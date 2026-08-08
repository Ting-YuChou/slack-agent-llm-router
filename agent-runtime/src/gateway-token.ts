import { createHmac, timingSafeEqual } from "node:crypto";

import {
  resolveAgentModel,
  type AgentModelApi,
  type AgentProvider,
} from "./agent-model.js";

export interface GatewayClaims {
  runId: string;
  provider: AgentProvider;
  model: string;
  api: AgentModelApi;
  reasoningEffort: "max";
  expiresAt: number;
}

export function issueGatewayToken(claims: GatewayClaims, secret: string): string {
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
    !isRecord(claims) ||
    typeof claims.runId !== "string" || !claims.runId ||
    typeof claims.provider !== "string" || typeof claims.model !== "string" ||
    typeof claims.api !== "string" || claims.reasoningEffort !== "max" ||
    typeof claims.expiresAt !== "number" || claims.expiresAt < now
  ) return null;
  let model;
  try {
    model = resolveAgentModel(`${claims.provider}/${claims.model}`);
  } catch {
    return null;
  }
  if (model.api !== claims.api || model.reasoningEffort !== claims.reasoningEffort) return null;
  return {
    runId: claims.runId,
    provider: model.provider,
    model: model.id,
    api: model.api,
    reasoningEffort: model.reasoningEffort,
    expiresAt: claims.expiresAt,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
