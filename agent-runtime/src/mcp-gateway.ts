import { createServer, type IncomingMessage, type ServerResponse } from "node:http";

import { verifyMcpToken, type McpClaims } from "./mcp-token.js";
import type { McpServerId } from "./mcp-config.js";

const DEFAULT_MAX_REQUEST_BYTES = 256_000;
const DEFAULT_MAX_RESULT_BYTES = 1_048_576;
const DEFAULT_TIMEOUT_MS = 30_000;

export interface McpAuditEvent {
  server: McpServerId;
  tool: string;
  scope: string;
  latency_ms: number;
  success: boolean;
  error_code: string | null;
  result_bytes: number;
}

export interface McpGatewayOptions {
  server: McpServerId;
  allowedTools: readonly string[];
  signingSecret: string;
  upstreamUrl: string;
  timeoutMs?: number;
  maxRequestBytes?: number;
  maxResultBytes?: number;
  onAudit?: (event: McpAuditEvent) => void;
  upstreamHeaders?: () => Promise<Record<string, string>>;
  healthCheck?: () => Promise<boolean>;
  authorizeTool?: (tool: string, args: Record<string, unknown>, claims: McpClaims) => void;
  transformResponse?: (method: string, tool: string, value: Record<string, any>) => void;
}

export function createMcpGatewayServer(options: McpGatewayOptions) {
  if (!options.signingSecret) throw new Error("MCP gateway signing secret is required");
  const upstreamUrl = new URL(options.upstreamUrl);
  if (!/^https?:$/.test(upstreamUrl.protocol)) throw new Error("MCP upstream URL must use HTTP(S)");
  const sessionBindings = new Map<string, { identity: string; expiry: number }>();
  const callCounts = new Map<string, number>();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxRequestBytes = options.maxRequestBytes ?? DEFAULT_MAX_REQUEST_BYTES;
  const maxResultBytes = options.maxResultBytes ?? DEFAULT_MAX_RESULT_BYTES;

  return createServer(async (request, response) => {
    if (request.method === "GET" && request.url === "/healthz") {
      const healthy = await options.healthCheck?.().catch(() => false) ?? true;
      sendJson(response, healthy ? 200 : 503, { status: healthy ? "healthy" : "unhealthy", server: options.server });
      return;
    }
    if (request.url !== "/mcp" || !["POST", "GET", "DELETE"].includes(request.method ?? "")) {
      sendError(response, 404, "not_found", "Endpoint not found");
      return;
    }
    const claims = authenticate(request, options.signingSecret, options.server);
    if (!claims || claims.tools.some((tool) => !options.allowedTools.includes(tool))) {
      sendError(response, 401, "unauthorized", "A valid MCP token is required");
      return;
    }
    for (const [sessionId, binding] of sessionBindings) {
      if (binding.expiry < Date.now()) sessionBindings.delete(sessionId);
    }
    const identity = `${claims.session}:${claims.run}:${claims.slackUser}`;
    const requestedSession = singleHeader(request.headers["mcp-session-id"]);
    if (requestedSession && sessionBindings.get(requestedSession)?.identity !== identity) {
      sendError(response, 403, "session_forbidden", "MCP session is not bound to this run");
      return;
    }
    if (request.method !== "POST") {
      await proxyStreamingRequest(request, response, upstreamUrl, timeoutMs, options.upstreamHeaders);
      if (request.method === "DELETE" && requestedSession) sessionBindings.delete(requestedSession);
      return;
    }

    const started = Date.now();
    let method = "unknown";
    let tool = "unknown";
    try {
      const raw = await readBody(request, maxRequestBytes);
      const message = parseJsonRpcRequest(raw);
      method = message.method;
      tool = method === "tools/call" && isRecord(message.params) && typeof message.params.name === "string"
        ? message.params.name : method;
      authorizeMessage(message, claims, options.authorizeTool);
      if (message.method === "tools/call") {
        const used = callCounts.get(identity) ?? 0;
        if (used >= claims.maxCalls) throw new GatewayRequestError(429, "call_budget_exceeded", "MCP call budget is exhausted");
        callCounts.set(identity, used + 1);
      }

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      let upstream: Response;
      try {
        const headers = upstreamHeaders(request);
        for (const [name, value] of Object.entries(await options.upstreamHeaders?.() ?? {})) headers.set(name, value);
        upstream = await fetch(upstreamUrl, {
          method: "POST",
          headers,
          body: raw.toString("utf8"),
          signal: controller.signal,
        });
      } catch (error) {
        clearTimeout(timer);
        const code = error instanceof Error && error.name === "AbortError" ? "upstream_timeout" : "upstream_unavailable";
        audit(options, claims, tool, started, false, code, 0);
        sendError(response, 502, code, "MCP upstream is unavailable");
        return;
      }
      const contentLength = Number(upstream.headers.get("content-length") ?? "0");
      if (Number.isFinite(contentLength) && contentLength > maxResultBytes) {
        clearTimeout(timer);
        controller.abort();
        audit(options, claims, tool, started, false, "result_too_large", contentLength);
        sendError(response, 502, "result_too_large", "MCP result exceeded the size limit");
        return;
      }
      let bytes: Buffer;
      try {
        bytes = await readResponseBody(upstream, maxResultBytes);
      } catch (error) {
        const tooLarge = error instanceof ResultTooLargeError;
        const code = tooLarge ? "result_too_large" : error instanceof Error && error.name === "AbortError"
          ? "upstream_timeout" : "upstream_unavailable";
        audit(options, claims, tool, started, false, code, tooLarge ? error.bytes : 0);
        sendError(response, 502, code, tooLarge
          ? "MCP result exceeded the size limit" : "MCP upstream is unavailable");
        return;
      } finally {
        clearTimeout(timer);
      }
      const contentType = upstream.headers.get("content-type") ?? "";
      let output: Buffer;
      try {
        output = sanitizeUpstreamResponse(bytes, contentType, message.method, tool, new Set(claims.tools), options.transformResponse);
      } catch {
        audit(options, claims, tool, started, false, "invalid_upstream_response", bytes.length);
        sendError(response, 502, "invalid_upstream_response", "MCP upstream returned an invalid response");
        return;
      }
      const upstreamSession = upstream.headers.get("mcp-session-id");
      if (upstreamSession) {
        const existing = sessionBindings.get(upstreamSession);
        if (existing && existing.identity !== identity) {
          audit(options, claims, tool, started, false, "session_collision", output.length);
          sendError(response, 502, "session_collision", "MCP upstream returned an invalid session");
          return;
        }
        sessionBindings.set(upstreamSession, { identity, expiry: claims.expiry });
        response.setHeader("mcp-session-id", upstreamSession);
      }
      const protocolVersion = upstream.headers.get("mcp-protocol-version");
      if (protocolVersion) response.setHeader("mcp-protocol-version", protocolVersion);
      response.statusCode = upstream.status;
      response.setHeader("content-type", contentType);
      response.setHeader("content-length", output.length);
      response.end(output);
      audit(options, claims, tool, started, upstream.ok, upstream.ok ? undefined : "upstream_error", output.length);
    } catch (error) {
      const gatewayError = error instanceof GatewayRequestError ? error : new GatewayRequestError(400, "invalid_request", "Invalid MCP request");
      audit(options, claims, tool === "unknown" ? method : tool, started, false, gatewayError.code, 0);
      sendError(response, gatewayError.status, gatewayError.code, gatewayError.message);
    }
  });
}

class GatewayRequestError extends Error {
  constructor(public readonly status: number, public readonly code: string, message: string) { super(message); }
}

class ResultTooLargeError extends Error {
  constructor(public readonly bytes: number) { super("MCP result exceeded the size limit"); }
}

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: unknown;
  method: string;
  params?: unknown;
}

function parseJsonRpcRequest(raw: Buffer): JsonRpcRequest {
  let value: unknown;
  try { value = JSON.parse(raw.toString("utf8")); } catch { throw new GatewayRequestError(400, "invalid_request", "Request must be JSON"); }
  if (!isRecord(value) || value.jsonrpc !== "2.0" || typeof value.method !== "string") {
    throw new GatewayRequestError(400, "invalid_request", "Request must be JSON-RPC 2.0");
  }
  return value as unknown as JsonRpcRequest;
}

function authorizeMessage(message: JsonRpcRequest, claims: McpClaims, authorizeTool?: McpGatewayOptions["authorizeTool"]): void {
  const allowedMethods = new Set(["initialize", "notifications/initialized", "notifications/cancelled", "ping", "tools/list", "tools/call"]);
  if (!allowedMethods.has(message.method)) throw new GatewayRequestError(403, "method_forbidden", "MCP method is not allowed");
  if (message.method !== "tools/call") return;
  if (!isRecord(message.params) || typeof message.params.name !== "string" || !isRecord(message.params.arguments)) {
    throw new GatewayRequestError(400, "invalid_request", "Tool call is invalid");
  }
  if (!claims.tools.includes(message.params.name)) {
    throw new GatewayRequestError(403, "tool_forbidden", "MCP tool is not allowed");
  }
  try { authorizeTool?.(message.params.name, message.params.arguments, claims); }
  catch { throw new GatewayRequestError(403, "scope_forbidden", "MCP scope is not allowed"); }
}

function repositoryMatches(tool: string, args: Record<string, unknown>, repository: string): boolean {
  if (tool === "search_code") {
    if (typeof args.query !== "string") return false;
    const qualifiers = [...args.query.matchAll(/(?:^|\s)repo:([^\s]+)/g)].map((match) => match[1]);
    return qualifiers.length === 1 && qualifiers[0]?.toLowerCase() === repository.toLowerCase();
  }
  const [owner, repo] = repository.split("/");
  return typeof args.owner === "string" && typeof args.repo === "string" &&
    args.owner.toLowerCase() === owner?.toLowerCase() && args.repo.toLowerCase() === repo?.toLowerCase();
}

function sanitizeUpstreamResponse(bytes: Buffer, contentType: string, method: string, tool: string, tools: Set<string>, transform?: McpGatewayOptions["transformResponse"]): Buffer {
  if (contentType.includes("application/json")) {
    const value = JSON.parse(bytes.toString("utf8"));
    if (!isRecord(value) || value.jsonrpc !== "2.0") throw new Error("invalid JSON-RPC response");
    transform?.(method, tool, value);
    filterToolList(value, method, tools);
    return Buffer.from(JSON.stringify(value));
  }
  if (contentType.includes("text/event-stream")) {
    const lines = bytes.toString("utf8").split("\n").map((line) => {
      if (!line.startsWith("data:")) return line;
      const value = JSON.parse(line.slice(5).trim());
      if (!isRecord(value) || value.jsonrpc !== "2.0") throw new Error("invalid SSE JSON-RPC response");
      transform?.(method, tool, value);
      filterToolList(value, method, tools);
      return `data: ${JSON.stringify(value)}`;
    });
    return Buffer.from(lines.join("\n"));
  }
  throw new Error("invalid response content type");
}

function filterToolList(value: Record<string, unknown>, method: string, tools: Set<string>): void {
  if (method !== "tools/list" || !isRecord(value.result) || !Array.isArray(value.result.tools)) return;
  value.result.tools = value.result.tools.filter((tool) => isRecord(tool) && typeof tool.name === "string" && tools.has(tool.name));
}

async function proxyStreamingRequest(
  request: IncomingMessage,
  response: ServerResponse,
  upstreamUrl: URL,
  timeoutMs: number,
  additionalHeaders?: () => Promise<Record<string, string>>,
): Promise<void> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const headers = upstreamHeaders(request);
    for (const [name, value] of Object.entries(await additionalHeaders?.() ?? {})) headers.set(name, value);
    const upstream = await fetch(upstreamUrl, { method: request.method, headers, signal: controller.signal });
    response.statusCode = upstream.status;
    for (const name of ["content-type", "mcp-session-id", "mcp-protocol-version"]) {
      const value = upstream.headers.get(name);
      if (value) response.setHeader(name, value);
    }
    if (!upstream.body) { response.end(); return; }
    const reader = upstream.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      response.write(Buffer.from(value));
    }
    response.end();
  } catch {
    if (!response.headersSent) sendError(response, 502, "upstream_unavailable", "MCP upstream is unavailable");
    else response.destroy();
  } finally {
    clearTimeout(timer);
  }
}

function authenticate(request: IncomingMessage, secret: string, server: McpServerId): McpClaims | null {
  const header = singleHeader(request.headers.authorization);
  if (!header?.startsWith("Bearer ")) return null;
  return verifyMcpToken(header.slice(7), secret, server);
}

function upstreamHeaders(request: IncomingMessage): Headers {
  const headers = new Headers();
  for (const name of ["accept", "content-type", "mcp-session-id", "mcp-protocol-version"]) {
    const value = singleHeader(request.headers[name]);
    if (value) headers.set(name, value);
  }
  return headers;
}

function singleHeader(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

async function readBody(request: IncomingMessage, limit: number): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of request) {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += bytes.length;
    if (total > limit) throw new GatewayRequestError(413, "request_too_large", "MCP request exceeded the size limit");
    chunks.push(bytes);
  }
  return Buffer.concat(chunks);
}

async function readResponseBody(response: Response, limit: number): Promise<Buffer> {
  if (!response.body) return Buffer.alloc(0);
  const reader = response.body.getReader();
  const chunks: Buffer[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = Buffer.from(value);
    total += chunk.length;
    if (total > limit) {
      await reader.cancel().catch(() => undefined);
      throw new ResultTooLargeError(total);
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, total);
}

function audit(options: McpGatewayOptions, claims: McpClaims, tool: string, started: number, success: boolean, errorCode: string | undefined, resultBytes: number): void {
  options.onAudit?.({
    server: claims.server,
    tool,
    scope: Object.keys(claims.scope).sort().join(","),
    latency_ms: Math.max(0, Date.now() - started),
    success,
    error_code: errorCode ?? null,
    result_bytes: resultBytes,
  });
}

function sendJson(response: ServerResponse, status: number, body: unknown): void {
  const payload = Buffer.from(JSON.stringify(body));
  response.statusCode = status;
  response.setHeader("content-type", "application/json");
  response.setHeader("content-length", payload.length);
  response.end(payload);
}

function sendError(response: ServerResponse, status: number, code: string, message: string): void {
  sendJson(response, status, { error: { code, message } });
}

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
