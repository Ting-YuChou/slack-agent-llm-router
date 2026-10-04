import { createHmac, timingSafeEqual } from "node:crypto";

import {
  JEV_CLASSIFIER_MODEL,
  resolveAgentModel,
  supportsAgentEffort,
  type AgentModelApi,
  type AgentProvider,
  type AgentReasoningEffort,
} from "./agent-model.js";

export interface GatewayClaims {
  runId: string;
  provider: AgentProvider;
  model: string;
  api: AgentModelApi;
  reasoningEffort: AgentReasoningEffort;
  expiresAt: number;
}

export interface ClassifierGatewayClaims {
  kind: "classifier";
  runId: string;
  provider: typeof JEV_CLASSIFIER_MODEL.provider;
  model: typeof JEV_CLASSIFIER_MODEL.id;
  api: typeof JEV_CLASSIFIER_MODEL.api;
  maxCalls: number;
  expiresAt: number;
}

export function issueGatewayToken(claims: GatewayClaims, secret: string): string {
  return signClaims(claims, secret);
}

export function issueClassifierGatewayToken(claims: ClassifierGatewayClaims, secret: string): string {
  return signClaims(claims, secret);
}

function signClaims(claims: GatewayClaims | ClassifierGatewayClaims, secret: string): string {
  if (!secret) throw new Error("Gateway signing secret is required");
  const payload = Buffer.from(JSON.stringify(claims)).toString("base64url");
  const signature = createHmac("sha256", secret).update(payload).digest("base64url");
  return `${payload}.${signature}`;
}

export function verifyClassifierGatewayToken(token: string, secret: string, now = Date.now()): ClassifierGatewayClaims | null {
  const claims = verifySignedClaims(token, secret);
  if (
    !claims ||
    claims.kind !== "classifier" ||
    typeof claims.runId !== "string" || !claims.runId ||
    claims.provider !== JEV_CLASSIFIER_MODEL.provider ||
    claims.model !== JEV_CLASSIFIER_MODEL.id ||
    claims.api !== JEV_CLASSIFIER_MODEL.api ||
    typeof claims.maxCalls !== "number" || !Number.isInteger(claims.maxCalls) || claims.maxCalls < 1 ||
    typeof claims.expiresAt !== "number" || claims.expiresAt < now
  ) return null;
  return {
    kind: "classifier",
    runId: claims.runId,
    provider: JEV_CLASSIFIER_MODEL.provider,
    model: JEV_CLASSIFIER_MODEL.id,
    api: JEV_CLASSIFIER_MODEL.api,
    maxCalls: claims.maxCalls,
    expiresAt: claims.expiresAt,
  };
}

export function verifyGatewayToken(token: string, secret: string, now = Date.now()): GatewayClaims | null {
  const claims = verifySignedClaims(token, secret);
  if (!claims) return null;
  if (
    typeof claims.runId !== "string" || !claims.runId ||
    typeof claims.provider !== "string" || typeof claims.model !== "string" ||
    typeof claims.api !== "string" || typeof claims.reasoningEffort !== "string" ||
    typeof claims.expiresAt !== "number" || claims.expiresAt < now
  ) return null;
  let model;
  try {
    model = resolveAgentModel(`${claims.provider}/${claims.model}`);
  } catch {
    return null;
  }
  if (model.api !== claims.api || !supportsAgentEffort(model, claims.reasoningEffort)) return null;
  return {
    runId: claims.runId,
    provider: model.provider,
    model: model.id,
    api: model.api,
    reasoningEffort: claims.reasoningEffort,
    expiresAt: claims.expiresAt,
  };
}

function verifySignedClaims(token: string, secret: string): Record<string, unknown> | null {
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
  return isRecord(claims) ? claims : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
