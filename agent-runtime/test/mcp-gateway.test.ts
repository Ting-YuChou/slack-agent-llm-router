import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer, type RequestListener, type Server } from "node:http";
import { test } from "node:test";

import { createMcpGatewayServer, MCP_SESSION_MESSAGE_ALLOWANCE, McpCallLedger, type McpAuditEvent } from "../src/mcp-gateway.js";
import { issueMcpToken, type McpClaims } from "../src/mcp-token.js";
import { authorizeGitHubTool } from "../src/mcp-policy.js";

const secret = "mcp-test-secret";
const baseClaims: McpClaims = {
  version: 2, run: "run-1", session: "session-1", slackUser: "U1",
  scope: { repository: "acme/widgets" }, server: "github",
  tools: ["get_file_contents", "search_code"], mode: "read_only", maxCalls: 12, expiry: Date.now() + 60_000,
};

const gatewayPolicy = { server: "github" as const, allowedTools: baseClaims.tools, authorizeTool: authorizeGitHubTool };

async function listen(server: Server): Promise<string> {
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("missing address");
  return `http://127.0.0.1:${address.port}`;
}

async function close(server: Server): Promise<void> {
  await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}

function token(claims: McpClaims = baseClaims): string {
  return issueMcpToken(claims, secret);
}

async function post(url: string, body: unknown, bearer = token(), headers: Record<string, string> = {}) {
  return fetch(`${url}/mcp`, {
    method: "POST",
    headers: { authorization: `Bearer ${bearer}`, "content-type": "application/json", accept: "application/json, text/event-stream", ...headers },
    body: JSON.stringify(body),
  });
}

test("gateway exposes health and replaces the run token with fixed GitHub upstream headers", async (t) => {
  let forwardedAuthorization: string | undefined;
  let forwardedReadonly: string | undefined;
  const upstream = createServer((request, response) => {
    forwardedAuthorization = request.headers.authorization;
    forwardedReadonly = request.headers["x-mcp-readonly"] as string | undefined;
    response.setHeader("content-type", "application/json");
    response.setHeader("mcp-session-id", "upstream-session");
    response.end(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { protocolVersion: "2025-06-18", capabilities: {}, serverInfo: { name: "github", version: "test" } } }));
  });
  const upstreamUrl = await listen(upstream);
  const gateway = createMcpGatewayServer({
    ...gatewayPolicy,
    signingSecret: secret,
    upstreamUrl: `${upstreamUrl}/mcp`,
    upstreamHeaders: async () => ({
      Authorization: "Bearer github-installation-token",
      "X-MCP-Readonly": "true",
      "X-MCP-Lockdown": "true",
      "X-MCP-Toolsets": "repos,issues,pull_requests",
      "X-MCP-Tools": baseClaims.tools.join(","),
    }),
  });
  const gatewayUrl = await listen(gateway);
  t.after(async () => { await close(gateway); await close(upstream); });

  const health = await fetch(`${gatewayUrl}/healthz`);
  assert.equal(health.status, 200);
  assert.deepEqual(await health.json(), { status: "healthy", server: "github" });
  const response = await post(gatewayUrl, { jsonrpc: "2.0", id: 1, method: "initialize", params: {} });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("mcp-session-id"), "upstream-session");
  assert.equal(forwardedAuthorization, "Bearer github-installation-token");
  assert.equal(forwardedReadonly, "true");
});

test("gateway health fails when GitHub token or upstream readiness fails", async (t) => {
  const gateway = createMcpGatewayServer({
    ...gatewayPolicy,
    signingSecret: secret,
    upstreamUrl: "http://127.0.0.1:1/mcp",
    healthCheck: async () => false,
  });
  const gatewayUrl = await listen(gateway);
  t.after(async () => { await close(gateway); });
  const health = await fetch(`${gatewayUrl}/healthz`);
  assert.equal(health.status, 503);
  assert.deepEqual(await health.json(), { status: "unhealthy", server: "github" });
});

test("gateway filters tools/list and audits only sanitized metadata", async (t) => {
  const upstream = createServer((_request, response) => {
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify({ jsonrpc: "2.0", id: 2, result: { tools: [
      { name: "get_file_contents", description: "allowed" },
      { name: "create_issue", description: "write" },
    ] } }));
  });
  const audits: McpAuditEvent[] = [];
  const upstreamUrl = await listen(upstream);
  const gateway = createMcpGatewayServer({ ...gatewayPolicy, signingSecret: secret, upstreamUrl: `${upstreamUrl}/mcp`, onAudit: (event) => audits.push(event) });
  const gatewayUrl = await listen(gateway);
  t.after(async () => { await close(gateway); await close(upstream); });

  const response = await post(gatewayUrl, { jsonrpc: "2.0", id: 2, method: "tools/list", params: {} });
  assert.equal(response.status, 200);
  assert.deepEqual((await response.json() as any).result.tools.map((tool: any) => tool.name), ["get_file_contents"]);
  assert.equal(audits.length, 1);
  assert.deepEqual(Object.keys(audits[0]).sort(), ["args_sha256", "error_code", "latency_ms", "result_bytes", "run", "scope", "server", "session", "success", "tool", "user"].sort());
  assert.equal(JSON.stringify(audits).includes("Bearer"), false);
  assert.equal(JSON.stringify(audits).includes("allowed"), false);
});

test("gateway enforces exact tool, repository, expiry and MCP session bindings", async (t) => {
  let upstreamCalls = 0;
  const upstream = createServer((_request, response) => {
    upstreamCalls += 1;
    response.setHeader("content-type", "application/json");
    response.setHeader("mcp-session-id", "bound-session");
    response.end(JSON.stringify({ jsonrpc: "2.0", id: 3, result: { content: [{ type: "text", text: "ok" }] } }));
  });
  const upstreamUrl = await listen(upstream);
  const gateway = createMcpGatewayServer({ ...gatewayPolicy, signingSecret: secret, upstreamUrl: `${upstreamUrl}/mcp` });
  const gatewayUrl = await listen(gateway);
  t.after(async () => { await close(gateway); await close(upstream); });

  const deniedTool = await post(gatewayUrl, { jsonrpc: "2.0", id: 3, method: "tools/call", params: { name: "create_issue", arguments: { owner: "acme", repo: "widgets" } } });
  assert.equal(deniedTool.status, 403);
  const wrongRepo = await post(gatewayUrl, { jsonrpc: "2.0", id: 4, method: "tools/call", params: { name: "get_file_contents", arguments: { owner: "other", repo: "widgets", path: "README.md" } } });
  assert.equal(wrongRepo.status, 403);
  const expired = await post(gatewayUrl, { jsonrpc: "2.0", id: 5, method: "tools/list", params: {} }, token({ ...baseClaims, expiry: Date.now() - 1 }));
  assert.equal(expired.status, 401);
  const forgedWriteAllowlist = await post(gatewayUrl, { jsonrpc: "2.0", id: 5, method: "tools/list", params: {} }, token({ ...baseClaims, tools: ["create_issue"] }));
  assert.equal(forgedWriteAllowlist.status, 401);
  const allowed = await post(gatewayUrl, { jsonrpc: "2.0", id: 6, method: "tools/call", params: { name: "get_file_contents", arguments: { owner: "acme", repo: "widgets", path: "README.md" } } });
  assert.equal(allowed.status, 200);
  const crossSession = await post(gatewayUrl, { jsonrpc: "2.0", id: 7, method: "tools/list", params: {} }, token({ ...baseClaims, run: "run-2", session: "session-2" }), { "mcp-session-id": "bound-session" });
  assert.equal(crossSession.status, 403);
  assert.equal(upstreamCalls, 1);
});

test("gateway enforces the token call budget per run", async (t) => {
  const upstream = createServer((_request, response) => {
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { content: [] } }));
  });
  const upstreamUrl = await listen(upstream);
  const gateway = createMcpGatewayServer({ ...gatewayPolicy, signingSecret: secret, upstreamUrl: `${upstreamUrl}/mcp` });
  const gatewayUrl = await listen(gateway);
  t.after(async () => { await close(gateway); await close(upstream); });
  const limited = token({ ...baseClaims, maxCalls: 1 });
  const body = { jsonrpc: "2.0", id: 1, method: "tools/call", params: { name: "get_file_contents", arguments: { owner: "acme", repo: "widgets" } } };
  assert.equal((await post(gatewayUrl, body, limited)).status, 200);
  const denied = await post(gatewayUrl, body, limited);
  assert.equal(denied.status, 429);
  assert.equal((await denied.json() as any).error.code, "call_budget_exceeded");
});

test("gateway maps upstream timeout, invalid responses and oversized results", async (t) => {
  const cases: Array<{ handler: RequestListener; expected: string; options?: Record<string, number> }> = [
    { handler: () => {}, expected: "upstream_timeout", options: { timeoutMs: 10 } },
    { handler: (_request, response) => { response.setHeader("content-type", "application/json"); response.end("not-json"); }, expected: "invalid_upstream_response" },
    { handler: (_request, response) => { response.setHeader("content-type", "application/json"); response.end(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { content: [{ type: "text", text: "x".repeat(500) }] } })); }, expected: "result_too_large", options: { maxResultBytes: 100 } },
  ];
  for (const item of cases) {
    const upstream = createServer(item.handler);
    const upstreamUrl = await listen(upstream);
    const gateway = createMcpGatewayServer({ ...gatewayPolicy, signingSecret: secret, upstreamUrl: `${upstreamUrl}/mcp`, ...item.options });
    const gatewayUrl = await listen(gateway);
    const response = await post(gatewayUrl, { jsonrpc: "2.0", id: 1, method: "tools/list", params: {} });
    assert.equal(response.status, 502);
    assert.equal((await response.json() as any).error.code, item.expected);
    await close(gateway);
    await close(upstream);
  }
  t.after(() => {});
});

test("gateway rejects GET and DELETE with 405 without reaching upstream", async (t) => {
  let upstreamCalls = 0;
  const upstream = createServer((_request, response) => {
    upstreamCalls += 1;
    response.setHeader("content-type", "text/event-stream");
    response.end("data: {}\n\n");
  });
  const upstreamUrl = await listen(upstream);
  const gateway = createMcpGatewayServer({ ...gatewayPolicy, signingSecret: secret, upstreamUrl: `${upstreamUrl}/mcp` });
  const gatewayUrl = await listen(gateway);
  t.after(async () => { await close(gateway); await close(upstream); });

  for (const method of ["GET", "DELETE"]) {
    const variants: Array<Record<string, string>> = [{ authorization: `Bearer ${token()}`, "mcp-session-id": "any-session" }, {}];
    for (const headers of variants) {
      const response = await fetch(`${gatewayUrl}/mcp`, { method, headers: { accept: "text/event-stream", ...headers } });
      assert.equal(response.status, 405, method);
      assert.equal(response.headers.get("allow"), "POST");
      assert.equal((await response.json() as any).error.code, "method_not_allowed");
    }
  }
  assert.equal(upstreamCalls, 0);
});

test("gateway counts initialize and tools/list toward the per-token budget", async (t) => {
  let upstreamCalls = 0;
  const upstream = createServer((_request, response) => {
    upstreamCalls += 1;
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { tools: [] } }));
  });
  const upstreamUrl = await listen(upstream);
  const gateway = createMcpGatewayServer({ ...gatewayPolicy, signingSecret: secret, upstreamUrl: `${upstreamUrl}/mcp` });
  const gatewayUrl = await listen(gateway);
  t.after(async () => { await close(gateway); await close(upstream); });
  const limited = token({ ...baseClaims, maxCalls: 1 });
  const budget = 1 + MCP_SESSION_MESSAGE_ALLOWANCE;

  assert.equal((await post(gatewayUrl, { jsonrpc: "2.0", id: 1, method: "initialize", params: {} }, limited)).status, 200);
  for (let index = 1; index < budget; index += 1) {
    assert.equal((await post(gatewayUrl, { jsonrpc: "2.0", id: index + 1, method: "tools/list", params: {} }, limited)).status, 200, `message ${index + 1}`);
  }
  const deniedList = await post(gatewayUrl, { jsonrpc: "2.0", id: 99, method: "tools/list", params: {} }, limited);
  assert.equal(deniedList.status, 429);
  assert.equal((await deniedList.json() as any).error.code, "call_budget_exceeded");
  const deniedInitialize = await post(gatewayUrl, { jsonrpc: "2.0", id: 100, method: "initialize", params: {} }, limited);
  assert.equal(deniedInitialize.status, 429);
  const deniedCall = await post(gatewayUrl, { jsonrpc: "2.0", id: 101, method: "tools/call", params: { name: "get_file_contents", arguments: { owner: "acme", repo: "widgets" } } }, limited);
  assert.equal(deniedCall.status, 429);
  assert.equal(upstreamCalls, budget);
});

test("gateway prunes call counts for expired tokens", async (t) => {
  const upstream = createServer((_request, response) => {
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { tools: [] } }));
  });
  const upstreamUrl = await listen(upstream);
  const callLedger = new McpCallLedger();
  const gateway = createMcpGatewayServer({ ...gatewayPolicy, signingSecret: secret, upstreamUrl: `${upstreamUrl}/mcp`, callLedger });
  const gatewayUrl = await listen(gateway);
  t.after(async () => { await close(gateway); await close(upstream); });
  const list = { jsonrpc: "2.0", id: 1, method: "tools/list", params: {} };

  assert.equal((await post(gatewayUrl, list, token({ ...baseClaims, expiry: Date.now() + 100 }))).status, 200);
  assert.equal(callLedger.size, 1);
  await new Promise((resolve) => setTimeout(resolve, 150));
  assert.equal((await post(gatewayUrl, list, token({ ...baseClaims, run: "run-2", session: "session-2" }))).status, 200);
  assert.equal(callLedger.size, 1);
});

test("gateway audits run, session, user and an argument hash but never raw arguments", async (t) => {
  const upstream = createServer((_request, response) => {
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { content: [{ type: "text", text: "ok" }] } }));
  });
  const audits: McpAuditEvent[] = [];
  const upstreamUrl = await listen(upstream);
  const gateway = createMcpGatewayServer({ ...gatewayPolicy, signingSecret: secret, upstreamUrl: `${upstreamUrl}/mcp`, onAudit: (event) => audits.push(event) });
  const gatewayUrl = await listen(gateway);
  t.after(async () => { await close(gateway); await close(upstream); });
  const args = { owner: "acme", path: "private/raw-argument-marker.txt", repo: "widgets" };

  assert.equal((await post(gatewayUrl, { jsonrpc: "2.0", id: 1, method: "tools/call", params: { name: "get_file_contents", arguments: args } })).status, 200);
  assert.equal((await post(gatewayUrl, { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "get_file_contents", arguments: { ...args, owner: "other" } } })).status, 403);
  assert.equal(audits.length, 2);
  assert.equal(audits[0]!.run, "run-1");
  assert.equal(audits[0]!.session, "session-1");
  assert.equal(audits[0]!.user, "U1");
  assert.equal(audits[0]!.tool, "get_file_contents");
  assert.equal(audits[0]!.args_sha256, createHash("sha256").update(JSON.stringify(args)).digest("hex"));
  assert.match(audits[1]!.args_sha256 ?? "", /^[0-9a-f]{64}$/);
  assert.notEqual(audits[1]!.args_sha256, audits[0]!.args_sha256);
  assert.equal(audits[1]!.error_code, "scope_forbidden");
  assert.equal(JSON.stringify(audits).includes("raw-argument-marker"), false);
});
