import type { WriteApprovalMode } from "./container-runtime.js";

export type ApprovalMode = "delivery" | "write";

export interface ApprovalSettings {
  mode: ApprovalMode;
  /** The owner approves the final diff before the orchestrator commits it. */
  deliveryApproval: boolean;
  /** What the writer's policy extension asks for on each workspace write. */
  writeApproval: WriteApprovalMode;
  deliveryTtlMs: number;
}

export const DEFAULT_DELIVERY_TTL_MS = 30 * 60_000;
export const MAX_DELIVERY_TTL_MS = 24 * 60 * 60_000;

/**
 * Reads PI_AGENT_APPROVAL_MODE (default `delivery`) and PI_AGENT_DELIVERY_TTL_MS.
 * Every result keeps at least one human gate: writes are approved one by one, or the diff is approved before commit.
 */
export function approvalSettings(env: Record<string, string | undefined>): ApprovalSettings {
  const raw = env.PI_AGENT_APPROVAL_MODE?.trim() || "delivery";
  if (raw !== "delivery" && raw !== "write") throw new Error("PI_AGENT_APPROVAL_MODE must be delivery or write");
  const ttlText = env.PI_AGENT_DELIVERY_TTL_MS?.trim();
  const ttl = ttlText ? Number(ttlText) : DEFAULT_DELIVERY_TTL_MS;
  if (!Number.isSafeInteger(ttl) || ttl <= 0) throw new Error("PI_AGENT_DELIVERY_TTL_MS must be a positive integer");
  return raw === "delivery"
    ? { mode: raw, deliveryApproval: true, writeApproval: "none", deliveryTtlMs: Math.min(ttl, MAX_DELIVERY_TTL_MS) }
    : { mode: raw, deliveryApproval: false, writeApproval: "per_prompt", deliveryTtlMs: Math.min(ttl, MAX_DELIVERY_TTL_MS) };
}
