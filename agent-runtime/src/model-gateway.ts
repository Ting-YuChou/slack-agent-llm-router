import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { Readable } from "node:stream";
import { pathToFileURL } from "node:url";

import { AGENT_MODEL_ID, AGENT_REASONING_EFFORT } from "./agent-model.js";
import { verifyGatewayToken } from "./gateway-token.js";

const MAX_BODY_BYTES = 2 * 1024 * 1024;
const OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses";

export function createModelGateway(options: {
  signingSecret: string;
  openaiApiKey: string;
  fetchFn?: typeof fetch;
}) {
  const fetchFn = options.fetchFn ?? fetch;
  return createServer(async (request, response) => {
    if (request.method === "GET" && request.url === "/health") {
      sendJson(response, 200, {
        status: "healthy",
        model: AGENT_MODEL_ID,
        reasoning_effort: AGENT_REASONING_EFFORT,
      });
      return;
    }
    if (request.method !== "POST" || request.url !== "/v1/responses") {
      sendJson(response, 404, { error: { code: "not_found", message: "Endpoint not found" } });
      return;
    }
    const authorization = request.headers.authorization ?? "";
    const token = authorization.startsWith("Bearer ") ? authorization.slice(7) : "";
    const claims = verifyGatewayToken(token, options.signingSecret);
    if (!claims) {
      sendJson(response, 401, { error: { code: "invalid_gateway_token", message: "Invalid gateway token" } });
      return;
    }
    let rawBody: Buffer;
    try {
      rawBody = await readBody(request);
      const body = JSON.parse(rawBody.toString("utf8"));
      if (!validateAgentModelRequest(body)) throw new Error("invalid model request");
    } catch {
      sendJson(response, 400, {
        error: {
          code: "invalid_request",
          message: `Only ${AGENT_MODEL_ID} Responses requests with ${AGENT_REASONING_EFFORT} reasoning are allowed`,
        },
      });
      return;
    }
    try {
      const upstream = await fetchFn(OPENAI_RESPONSES_URL, {
        method: "POST",
        headers: {
          authorization: `Bearer ${options.openaiApiKey}`,
          "content-type": "application/json",
          "x-pi-run-id": claims.runId,
        },
        body: rawBody.toString("utf8"),
      });
      const headers: Record<string, string> = { "cache-control": "no-store" };
      for (const name of ["content-type", "openai-request-id", "x-request-id"]) {
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

export function validateAgentModelRequest(body: unknown): boolean {
  if (!isRecord(body) || body.model !== AGENT_MODEL_ID) return false;
  const reasoning = body.reasoning;
  return isRecord(reasoning) && reasoning.effort === AGENT_REASONING_EFFORT;
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
  const openaiApiKey = process.env.OPENAI_API_KEY ?? "";
  if (!signingSecret || !openaiApiKey) throw new Error("MODEL_GATEWAY_SIGNING_SECRET and OPENAI_API_KEY are required");
  createModelGateway({ signingSecret, openaiApiKey }).listen(8080, "0.0.0.0");
}
