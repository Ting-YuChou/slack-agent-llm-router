import { createHmac, timingSafeEqual } from "node:crypto";

import { AGENT_MODEL_ID } from "./agent-model.js";

export interface GatewayClaims {
  runId: string;
  model: typeof AGENT_MODEL_ID;
  expiresAt: number;
}

export function issueGatewayToken(claims: { runId: string; model: string; expiresAt: number }, secret: string): string {
  if (!secret) throw new Error("Gateway signing secret is required");
  const payload = Buffer.from(JSON.stringify(claims)).toString("base64url");
  const signature = createHmac("sha256", secret).update(payload).digest("base64url");
  return `${payload}.${signature}`;
}

export function verifyGatewayToken(token: string, secret: string, now = Date.now()): GatewayClaims | null {
  const [payload, signature, extra] = token.split(".");
  if (!payload || !signature || extra || !secret) return null;
  const expected = createHmac("sha256", secret).update(payload).digest();
  let provided: Buffer;
  try {
    provided = Buffer.from(signature, "base64url");
  } catch {
    return null;
  }
  if (provided.length !== expected.length || !timingSafeEqual(provided, expected)) return null;
  let claims: unknown;
  try {
    claims = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
  } catch {
    return null;
  }
  if (
    !isRecord(claims) || claims.model !== AGENT_MODEL_ID ||
    typeof claims.runId !== "string" || !claims.runId ||
    typeof claims.expiresAt !== "number" || claims.expiresAt < now
  ) return null;
  return { runId: claims.runId, model: AGENT_MODEL_ID, expiresAt: claims.expiresAt };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
