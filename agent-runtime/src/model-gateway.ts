import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { Readable } from "node:stream";
import { pathToFileURL } from "node:url";

import {
  JEV_CLASSIFIER_MODEL,
  listAgentModels,
  resolveAgentModel,
  type AgentModelSpec,
  type AgentProvider,
  type AgentReasoningEffort,
} from "./agent-model.js";
import { verifyClassifierGatewayToken, verifyGatewayToken } from "./gateway-token.js";

const MAX_BODY_BYTES = 2 * 1024 * 1024;

export type ProviderApiKeys = Partial<Record<AgentProvider | "openrouter", string>>;

export function createModelGateway(options: {
  signingSecret: string;
  providerApiKeys: ProviderApiKeys;
  fetchFn?: typeof fetch;
  classifierTimeoutMs?: number;
}) {
  const fetchFn = options.fetchFn ?? fetch;
  const classifierCalls = new Map<string, { count: number; expiresAt: number }>();
  return createServer(async (request, response) => {
    if (request.method === "GET" && request.url === "/health") {
      sendJson(response, 200, {
        status: "healthy",
        providers: listAgentModels().map((model) => ({
          provider: model.provider,
          model: model.id,
          configured: Boolean(options.providerApiKeys[model.provider]),
        })),
      });
      return;
    }
    if (request.method !== "POST") {
      sendJson(response, 404, { error: { code: "not_found", message: "Endpoint not found" } });
      return;
    }
    const token = extractToken(request);
    if (request.url === JEV_CLASSIFIER_MODEL.gatewayPath) {
      const claims = verifyClassifierGatewayToken(token, options.signingSecret);
      if (!claims) {
        sendJson(response, 401, { error: { code: "invalid_gateway_token", message: "Invalid gateway token" } });
        return;
      }
      const providerApiKey = options.providerApiKeys.openrouter;
      if (!providerApiKey) {
        sendJson(response, 503, { error: { code: "provider_not_configured", message: "Requested model provider is not configured" } });
        return;
      }
      let rawBody: Buffer;
      try {
        rawBody = await readBody(request);
        if (rawBody.length > 128 * 1024 || !validateClassifierModelRequest(JSON.parse(rawBody.toString("utf8")))) {
          throw new Error("invalid classifier request");
        }
      } catch {
        sendJson(response, 400, {
          error: { code: "invalid_request", message: "Request does not match the token-bound classifier configuration" },
        });
        return;
      }
      const now = Date.now();
      for (const [runId, entry] of classifierCalls) {
        if (entry.expiresAt < now) classifierCalls.delete(runId);
      }
      const budget = classifierCalls.get(claims.runId) ?? { count: 0, expiresAt: claims.expiresAt };
      if (budget.count >= claims.maxCalls) {
        sendJson(response, 429, {
          error: { code: "classifier_call_budget_exhausted", message: "Classifier call budget exhausted" },
        });
        return;
      }
      classifierCalls.set(claims.runId, { count: budget.count + 1, expiresAt: claims.expiresAt });
      try {
        const upstream = await fetchFn(JEV_CLASSIFIER_MODEL.upstreamUrl, {
          method: "POST",
          headers: {
            authorization: `Bearer ${providerApiKey}`,
            "content-type": "application/json",
            "x-pi-run-id": claims.runId,
          },
          body: rawBody.toString("utf8"),
          signal: AbortSignal.timeout(options.classifierTimeoutMs ?? 5_000),
        });
        const headers: Record<string, string> = { "cache-control": "no-store" };
        for (const name of ["content-type", "request-id", "x-request-id"]) {
          const value = upstream.headers.get(name);
          if (value) headers[name] = value;
        }
        response.writeHead(upstream.status, headers);
        if (upstream.body) Readable.fromWeb(upstream.body as never).pipe(response);
        else response.end();
      } catch {
        sendJson(response, 502, { error: { code: "upstream_unavailable", message: "Model provider unavailable" } });
      }
      return;
    }
    const claims = verifyGatewayToken(token, options.signingSecret);
    if (!claims) {
      sendJson(response, 401, { error: { code: "invalid_gateway_token", message: "Invalid gateway token" } });
      return;
    }
    const model = resolveAgentModel(`${claims.provider}/${claims.model}`);
    if (request.url !== model.gatewayPath) {
      sendJson(response, 403, { error: { code: "provider_path_mismatch", message: "Gateway token cannot use this provider path" } });
      return;
    }
    const providerApiKey = options.providerApiKeys[model.provider];
    if (!providerApiKey) {
      sendJson(response, 503, { error: { code: "provider_not_configured", message: "Requested model provider is not configured" } });
      return;
    }
    let rawBody: Buffer;
    try {
      rawBody = await readBody(request);
      const body = JSON.parse(rawBody.toString("utf8"));
      if (!validateAgentModelRequest(body, model, claims.reasoningEffort)) throw new Error("invalid model request");
    } catch {
      sendJson(response, 400, {
        error: { code: "invalid_request", message: "Request does not match the token-bound model configuration" },
      });
      return;
    }
    try {
      const upstream = await fetchFn(model.upstreamUrl, {
        method: "POST",
        headers: upstreamHeaders(request, model, providerApiKey, claims.runId),
        body: rawBody.toString("utf8"),
      });
      const headers: Record<string, string> = { "cache-control": "no-store" };
      for (const name of ["content-type", "openai-request-id", "request-id", "x-request-id"]) {
        const value = upstream.headers.get(name);
        if (value) headers[name] = value;
      }
      response.writeHead(upstream.status, headers);
      if (upstream.body) Readable.fromWeb(upstream.body as never).pipe(response);
      else response.end();
    } catch {
      sendJson(response, 502, { error: { code: "upstream_unavailable", message: "Model provider unavailable" } });
    }
  });
}

export function validateAgentModelRequest(body: unknown, model: AgentModelSpec, effort: AgentReasoningEffort = model.reasoningEffort): boolean {
  if (!isRecord(body) || body.model !== model.id) return false;
  if (model.api === "openai-responses") {
    return isRecord(body.reasoning) && body.reasoning.effort === effort;
  }
  if (model.api === "anthropic-messages") {
    return isRecord(body.output_config) && body.output_config.effort === effort;
  }
  return isRecord(body.thinking)
    && body.thinking.type === "enabled"
    && body.reasoning_effort === effort;
}

export function validateClassifierModelRequest(body: unknown): boolean {
  if (!isRecord(body) || body.model !== JEV_CLASSIFIER_MODEL.id || !isRecord(body.state) || !isRecord(body.questions)) {
    return false;
  }
  const questions = Object.entries(body.questions);
  if (questions.length < 1 || questions.length > 16) return false;
  return questions.every(([id, value]) => {
    if (!id || id.length > 100 || !isRecord(value) || !boundedText(value.instructions, 2_000)) return false;
    if (value.type === "choice") {
      if (!isRecord(value.criteria)) return false;
      const criteria = Object.entries(value.criteria);
      return criteria.length >= 2 && criteria.length <= 32
        && criteria.every(([label, description]) => Boolean(label) && label.length <= 100 && boundedText(description, 2_000));
    }
    if (value.type === "score") {
      return Array.isArray(value.criteria) && value.criteria.length >= 2 && value.criteria.length <= 16
        && value.criteria.every((description) => boundedText(description, 2_000));
    }
    if (value.type === "noul") {
      return isRecord(value.criteria)
        && boundedText(value.criteria.true, 2_000)
        && boundedText(value.criteria.false, 2_000);
    }
    return false;
  });
}

function boundedText(value: unknown, maxLength: number): value is string {
  return typeof value === "string" && value.trim().length > 0 && value.length <= maxLength;
}

function extractToken(request: IncomingMessage): string {
  const authorization = request.headers.authorization ?? "";
  if (authorization.startsWith("Bearer ")) return authorization.slice(7);
  const apiKey = request.headers["x-api-key"];
  return typeof apiKey === "string" ? apiKey : "";
}

function upstreamHeaders(
  request: IncomingMessage,
  model: AgentModelSpec,
  apiKey: string,
  runId: string,
): Record<string, string> {
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "x-pi-run-id": runId,
  };
  for (const name of ["accept", "anthropic-version", "anthropic-beta"]) {
    const value = request.headers[name];
    if (typeof value === "string") headers[name] = value;
  }
  if (model.provider === "anthropic") headers["x-api-key"] = apiKey;
  else headers.authorization = `Bearer ${apiKey}`;
  return headers;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function readBody(request: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buffer.length;
    if (total > MAX_BODY_BYTES) throw new Error("request too large");
    chunks.push(buffer);
  }
  return Buffer.concat(chunks);
}

function sendJson(response: ServerResponse, status: number, payload: unknown): void {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
  response.end(JSON.stringify(payload));
}

const entrypoint = process.argv[1] ? pathToFileURL(process.argv[1]).href : undefined;
if (entrypoint === import.meta.url) {
  const signingSecret = process.env.MODEL_GATEWAY_SIGNING_SECRET ?? "";
  const providerApiKeys: ProviderApiKeys = {
    openai: process.env.OPENAI_API_KEY,
    anthropic: process.env.ANTHROPIC_API_KEY,
    "opencode-go": process.env.OPENCODE_API_KEY,
    openrouter: process.env.OPENROUTER_API_KEY,
  };
  if (!signingSecret || !Object.values(providerApiKeys).some(Boolean)) {
    throw new Error("MODEL_GATEWAY_SIGNING_SECRET and at least one provider API key are required");
  }
  createModelGateway({ signingSecret, providerApiKeys }).listen(8080, "0.0.0.0");
}
