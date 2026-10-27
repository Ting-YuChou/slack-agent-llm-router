import { createHash } from "node:crypto";

/** Session entry the container's policy extensions append for each decision; the host only counts them. */
export const POLICY_ENTRY = "pi_agent_policy";

export type PolicyVerdict = "allow" | "block" | "approval";
export interface PolicyDecision {
  tool: string; decision: PolicyVerdict; reason: string;
  /** Only for approvals: what the owner answered. */
  approved?: boolean;
  /** Basename of the command's first word, never the command itself. */
  argv0?: string;
  /** Prefix of the command's sha256, so repeats can be counted without storing the text. */
  cmd_hash?: string;
  latency_ms?: number;
}
export interface PolicySummary { allow: number; block: number; approval: number; approved: number; by_reason: Record<string, number> }

const CODE = /^[a-z][a-z0-9_]{0,63}$/;
const ARGV0 = /^[\w.+-]{1,64}$/;
const HASH = /^[0-9a-f]{16}$/;
const VERDICTS = new Set<PolicyVerdict>(["allow", "block", "approval"]);
/** Distinct reasons kept per run; the rest are counted as `other`. */
const MAX_REASONS = 32;

function code(text: string): string {
  const normalized = text.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 64).replace(/_+$/, "");
  return CODE.test(normalized) ? normalized : "unspecified";
}

/** Builds the entry an extension appends: bounded codes, the command reduced to its basename and a hash. */
export function policyDecisionEntry(input: { tool: string; decision: PolicyVerdict; reason: string; approved?: boolean; command?: string; latencyMs?: number }): PolicyDecision {
  const entry: PolicyDecision = { tool: code(input.tool), decision: input.decision, reason: code(input.reason) };
  if (input.decision === "approval" && input.approved !== undefined) entry.approved = input.approved;
  const first = input.command?.trim().split(/\s+/)[0];
  if (first) {
    const base = first.split("/").pop() ?? "";
    if (ARGV0.test(base)) entry.argv0 = base;
    entry.cmd_hash = createHash("sha256").update(input.command!).digest("hex").slice(0, 16);
  }
  if (input.latencyMs !== undefined && Number.isFinite(input.latencyMs) && input.latencyMs >= 0) entry.latency_ms = Math.round(input.latencyMs);
  return entry;
}

/** Appends a decision entry for the host; failing to record it never changes the decision. */
export function appendPolicyDecision(pi: { appendEntry(customType: string, data: unknown): void }, input: Parameters<typeof policyDecisionEntry>[0]): void {
  try { pi.appendEntry(POLICY_ENTRY, policyDecisionEntry(input)); } catch { /* monitoring must not change a decision */ }
}

/** Host-side check of an entry from the container: anything malformed or unexpected is dropped, never trusted. */
export function parsePolicyDecision(data: unknown): PolicyDecision | undefined {
  if (!data || typeof data !== "object" || Array.isArray(data)) return undefined;
  const value = data as Record<string, unknown>;
  if (typeof value.tool !== "string" || !CODE.test(value.tool)) return undefined;
  if (typeof value.decision !== "string" || !VERDICTS.has(value.decision as PolicyVerdict)) return undefined;
  if (typeof value.reason !== "string" || !CODE.test(value.reason)) return undefined;
  const parsed: PolicyDecision = { tool: value.tool, decision: value.decision as PolicyVerdict, reason: value.reason };
  if (value.approved !== undefined) {
    if (typeof value.approved !== "boolean" || parsed.decision !== "approval") return undefined;
    parsed.approved = value.approved;
  }
  if (value.argv0 !== undefined) { if (typeof value.argv0 !== "string" || !ARGV0.test(value.argv0)) return undefined; parsed.argv0 = value.argv0; }
  if (value.cmd_hash !== undefined) { if (typeof value.cmd_hash !== "string" || !HASH.test(value.cmd_hash)) return undefined; parsed.cmd_hash = value.cmd_hash; }
  if (value.latency_ms !== undefined) {
    if (typeof value.latency_ms !== "number" || !Number.isInteger(value.latency_ms) || value.latency_ms < 0 || value.latency_ms > 86_400_000) return undefined;
    parsed.latency_ms = value.latency_ms;
  }
  return parsed;
}

export function summarizePolicy(summary: PolicySummary | undefined, decision: PolicyDecision): PolicySummary {
  const next = summary ?? { allow: 0, block: 0, approval: 0, approved: 0, by_reason: {} };
  next[decision.decision] += 1;
  if (decision.approved) next.approved += 1;
  const known = decision.reason in next.by_reason || Object.keys(next.by_reason).filter(key => key !== "other").length < MAX_REASONS;
  const key = known ? decision.reason : "other";
  next.by_reason[key] = (next.by_reason[key] ?? 0) + 1;
  return next;
}
