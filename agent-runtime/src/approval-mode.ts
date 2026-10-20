import type { WriteApprovalMode } from "./container-runtime.js";

export type ApprovalMode = "delivery";

export interface ApprovalSettings {
  mode: ApprovalMode;
  /** The owner approves the final diff before the orchestrator commits it. */
  deliveryApproval: true;
  /** What the writer's policy extension asks for on each workspace write; delivery mode approves the diff instead. */
  writeApproval: WriteApprovalMode;
  deliveryTtlMs: number;
}

export const DEFAULT_DELIVERY_TTL_MS = 30 * 60_000;
export const MAX_DELIVERY_TTL_MS = 24 * 60 * 60_000;

/**
 * Reads PI_AGENT_APPROVAL_MODE (only `delivery`, the default) and PI_AGENT_DELIVERY_TTL_MS.
 * Every run keeps a human gate: the owner approves the diff before commit. CI repair sessions skip
 * delivery approval and keep per-write prompts instead (see server.ts).
 */
export function approvalSettings(env: Record<string, string | undefined>): ApprovalSettings {
  const raw = env.PI_AGENT_APPROVAL_MODE?.trim() || "delivery";
  if (raw.toLowerCase() === "write") {
    throw new Error("PI_AGENT_APPROVAL_MODE=write was removed; set PI_AGENT_APPROVAL_MODE=delivery (or unset it) so the owner approves the final diff. See docs/agent-verification-ci.md");
  }
  if (raw !== "delivery") throw new Error("PI_AGENT_APPROVAL_MODE must be delivery");
  const ttlText = env.PI_AGENT_DELIVERY_TTL_MS?.trim();
  const ttl = ttlText ? Number(ttlText) : DEFAULT_DELIVERY_TTL_MS;
  if (!Number.isSafeInteger(ttl) || ttl <= 0) throw new Error("PI_AGENT_DELIVERY_TTL_MS must be a positive integer");
  return { mode: raw, deliveryApproval: true, writeApproval: "none", deliveryTtlMs: Math.min(ttl, MAX_DELIVERY_TTL_MS) };
}
