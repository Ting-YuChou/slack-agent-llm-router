import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { guardedHandler } from "./http-handler.js";

import { verifyMcpToken, type McpClaims } from "./mcp-token.js";
import type { McpServerId } from "./mcp-config.js";

const DEFAULT_MAX_REQUEST_BYTES = 256_000;
const DEFAULT_MAX_RESULT_BYTES = 1_048_576;
const DEFAULT_TIMEOUT_MS = 30_000;
/**
 * Messages a token may send on top of its tool calls: two handshakes of initialize,
 * notifications/initialized and tools/list, so a client can reconnect once.
 */
export const MCP_SESSION_MESSAGE_ALLOWANCE = 6;

export interface McpAuditEvent {
  server: McpServerId;
  tool: string;
  scope: string;
  run: string;
  session: string;
  user: string;
  /** sha256 of the canonical JSON of the tool arguments (or params); the arguments are never logged. */
  args_sha256: string | null;
  latency_ms: number;
  success: boolean;
  error_code: string | null;
  result_bytes: number;
}

export interface McpGatewayOptions {
  /** Whether a token's execution belongs to a run that is still active; ended runs' tokens are refused. */
  isActive?: (run: string) => Promise<boolean>;
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
  callLedger?: McpCallLedger;
}

/** Per-token message counts. Every message counts; tool calls are also capped at the token's maxCalls. */
export class McpCallLedger {
  private readonly entries = new Map<string, { messages: number; toolCalls: number; expiry: number }>();

  get size(): number { return this.entries.size; }

  /** Records one message, or returns false when the token's budget is spent. */
  consume(identity: string, claims: McpClaims, toolCall: boolean): boolean {
    const entry = this.entries.get(identity) ?? { messages: 0, toolCalls: 0, expiry: claims.expiry };
    if (entry.messages >= claims.maxCalls + MCP_SESSION_MESSAGE_ALLOWANCE || (toolCall && entry.toolCalls >= claims.maxCalls)) return false;
    entry.messages += 1;
    if (toolCall) entry.toolCalls += 1;
    entry.expiry = Math.max(entry.expiry, claims.expiry);
    this.entries.set(identity, entry);
    return true;
  }

  /** Drops the counts of tokens that have expired; they can no longer authenticate. */
  prune(now = Date.now()): void {
    for (const [identity, entry] of this.entries) {
      if (entry.expiry < now) this.entries.delete(identity);
    }
  }
}

export function createMcpGatewayServer(options: McpGatewayOptions) {
  if (!options.signingSecret) throw new Error("MCP gateway signing secret is required");
  const upstreamUrl = new URL(options.upstreamUrl);
  if (!/^https?:$/.test(upstreamUrl.protocol)) throw new Error("MCP upstream URL must use HTTP(S)");
  const sessionBindings = new Map<string, { identity: string; expiry: number }>();
  const callLedger = options.callLedger ?? new McpCallLedger();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxRequestBytes = options.maxRequestBytes ?? DEFAULT_MAX_REQUEST_BYTES;
  const maxResultBytes = options.maxResultBytes ?? DEFAULT_MAX_RESULT_BYTES;

  return createServer(guardedHandler("mcp-gateway", async (request, response) => {
    if (request.method === "GET" && request.url === "/healthz") {
      const healthy = await options.healthCheck?.().catch(() => false) ?? true;
      sendJson(response, healthy ? 200 : 503, { status: healthy ? "healthy" : "unhealthy", server: options.server });
      return;
    }
    if (request.url !== "/mcp") {
      sendError(response, 404, "not_found", "Endpoint not found");
      return;
    }
    // GET (server-sent event stream) and DELETE would bypass request filtering, size limits and counting.
    if (request.method !== "POST") {
      response.setHeader("allow", "POST");
      sendError(response, 405, "method_not_allowed", "Only POST is supported");
      return;
    }
    const claims = authenticate(request, options.signingSecret, options.server);
    if (!claims || claims.tools.some((tool) => !options.allowedTools.includes(tool))) {
      sendError(response, 401, "unauthorized", "A valid MCP token is required");
      return;
    }
    if (options.isActive) {
      let active: boolean;
      try { active = await options.isActive(claims.run); }
      catch {
        sendError(response, 503, "revocation_unavailable", "Whether this run is still active could not be confirmed");
        return;
      }
      if (!active) {
        sendError(response, 401, "token_revoked", "This run has ended; its MCP token is revoked");
        return;
      }
    }
    for (const [sessionId, binding] of sessionBindings) {
      if (binding.expiry < Date.now()) sessionBindings.delete(sessionId);
    }
    callLedger.prune();
    const identity = `${claims.session}:${claims.run}:${claims.slackUser}`;
    const requestedSession = singleHeader(request.headers["mcp-session-id"]);
    if (requestedSession && sessionBindings.get(requestedSession)?.identity !== identity) {
      sendError(response, 403, "session_forbidden", "MCP session is not bound to this run");
      return;
    }

    const started = Date.now();
    let method = "unknown";
    let tool = "unknown";
    let argsSha256: string | null = null;
    const audit = (success: boolean, errorCode: string | undefined, resultBytes: number) =>
      recordAudit(options, claims, tool === "unknown" ? method : tool, argsSha256, started, success, errorCode, resultBytes);
    try {
      const raw = await readBody(request, maxRequestBytes);
      const message = parseJsonRpcRequest(raw);
      method = message.method;
      tool = method === "tools/call" && isRecord(message.params) && typeof message.params.name === "string"
        ? message.params.name : method;
      // Hashed before authorization, which may rewrite the arguments.
      argsSha256 = argumentsHash(message);
      authorizeMessage(message, claims, options.authorizeTool);
      if (!callLedger.consume(identity, claims, message.method === "tools/call")) {
        throw new GatewayRequestError(429, "call_budget_exceeded", "MCP call budget is exhausted");
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
        audit(false, code, 0);
        sendError(response, 502, code, "MCP upstream is unavailable");
        return;
      }
      const contentLength = Number(upstream.headers.get("content-length") ?? "0");
      if (Number.isFinite(contentLength) && contentLength > maxResultBytes) {
        clearTimeout(timer);
        controller.abort();
        audit(false, "result_too_large", contentLength);
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
        audit(false, code, tooLarge ? error.bytes : 0);
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
        audit(false, "invalid_upstream_response", bytes.length);
        sendError(response, 502, "invalid_upstream_response", "MCP upstream returned an invalid response");
        return;
      }
      const upstreamSession = upstream.headers.get("mcp-session-id");
      if (upstreamSession) {
        const existing = sessionBindings.get(upstreamSession);
        if (existing && existing.identity !== identity) {
          audit(false, "session_collision", output.length);
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
      audit(upstream.ok, upstream.ok ? undefined : "upstream_error", output.length);
    } catch (error) {
      const gatewayError = error instanceof GatewayRequestError ? error : new GatewayRequestError(400, "invalid_request", "Invalid MCP request");
      audit(false, gatewayError.code, 0);
      sendError(response, gatewayError.status, gatewayError.code, gatewayError.message);
    }
  }));
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

function argumentsHash(message: JsonRpcRequest): string | null {
  const value = message.method === "tools/call" && isRecord(message.params) ? message.params.arguments : message.params;
  return value === undefined ? null : createHash("sha256").update(canonicalJson(value)).digest("hex");
}

/** JSON with object keys sorted, so equal arguments hash equally whatever their key order. */
function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (isRecord(value)) return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  return JSON.stringify(value) ?? "null";
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

function recordAudit(
  options: McpGatewayOptions, claims: McpClaims, tool: string, argsSha256: string | null,
  started: number, success: boolean, errorCode: string | undefined, resultBytes: number,
): void {
  options.onAudit?.({
    server: claims.server,
    tool,
    scope: Object.keys(claims.scope).sort().join(","),
    run: claims.run,
    session: claims.session,
    user: claims.slackUser,
    args_sha256: argsSha256,
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
