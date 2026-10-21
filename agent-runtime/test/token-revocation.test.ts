import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { JEV_CLASSIFIER_MODEL } from "../src/agent-model.js";
import { BudgetLedger } from "../src/budget-ledger.js";
import { BudgetRpcClient } from "../src/budget-rpc.js";
import { deriveMcpRevocationSecret, ExecutionStatusClient, executionStatusFromEnv } from "../src/execution-status.js";
import { issueClassifierGatewayToken } from "../src/gateway-token.js";
import { createMcpGatewayServer } from "../src/mcp-gateway.js";
import { authorizeGitHubTool } from "../src/mcp-policy.js";
import { issueMcpToken, type McpClaims } from "../src/mcp-token.js";
import { createModelGateway } from "../src/model-gateway.js";
import { executionActive } from "../src/run-budget.js";
import { createAgentHttpServer } from "../src/server.js";

async function listen(server: Server): Promise<string> {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}
const close = (server: Server) => new Promise<void>((resolve) => server.close(() => resolve()));

test("a budget is active until it is stopped, and an unknown one never is", () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "revocation-ledger-"));
  const ledger = new BudgetLedger(path.join(root, "db"));
  try {
    ledger.open("run-1", 5);
    assert.equal(ledger.active("run-1"), true);
    ledger.stop("run-1");
    assert.equal(ledger.active("run-1"), false);
    assert.equal(ledger.active("never-opened"), false);
  } finally { ledger.close(); rmSync(root, { recursive: true, force: true }); }
});

test("an execution is active while the budget it spends is: its parent run's, or its own", () => {
  const active = new Set(["run-1", "solo"]);
  const ledger = { active: (id: string) => active.has(id) };
  const parents = new Map([["exec-a", "run-1"]]);
  assert.equal(executionActive("exec-a", parents, ledger), true);
  assert.equal(executionActive("solo", parents, ledger), true);
  // A restart loses the in-memory map; the control store's record still resolves the parent.
  assert.equal(executionActive("exec-b", parents, ledger, (id) => id === "exec-b" ? "run-1" : undefined), true);
  active.delete("run-1");
  assert.equal(executionActive("exec-a", parents, ledger), false);
  assert.equal(executionActive("exec-b", parents, ledger, () => "run-1"), false);
});

test("the host answers execution status to the model gateway and, with their own bearers, to MCP gateways", async () => {
  const mcpSecret = deriveMcpRevocationSecret("github-server-secret");
  const asked: string[] = [];
  const server = createAgentHttpServer({
    token: "runtime", orchestrator: {} as never, health: () => ({}), budgetSecret: "budget-bearer",
    budgetLedger: {} as never, executionActive: (execution: string) => { asked.push(execution); return execution === "live"; },
    mcpRevocationSecrets: [mcpSecret],
  } as never);
  const base = await listen(server);
  try {
    const budget = new BudgetRpcClient(base, "budget-bearer");
    assert.equal(await budget.active("live"), true);
    assert.equal(await budget.active("ended"), false);
    const mcp = new ExecutionStatusClient(base, mcpSecret);
    assert.equal(await mcp.active("live"), true);
    assert.equal(await mcp.active("ended"), false);
    assert.deepEqual(asked, ["live", "ended", "live", "ended"]);
    // Neither bearer opens the other's endpoint, and the raw server secret opens neither.
    await assert.rejects(new ExecutionStatusClient(base, "budget-bearer").active("live"));
    await assert.rejects(new BudgetRpcClient(base, mcpSecret).active("live"));
    await assert.rejects(new ExecutionStatusClient(base, "github-server-secret").active("live"));
  } finally { await close(server); }
});

test("the MCP revocation bearer is derived from, and never equal to, the server's signing secret", () => {
  const derived = deriveMcpRevocationSecret("github-server-secret");
  assert.notEqual(derived, "github-server-secret");
  assert.equal(derived, deriveMcpRevocationSecret("github-server-secret"));
  assert.notEqual(derived, deriveMcpRevocationSecret("context7-server-secret"));
  assert.throws(() => deriveMcpRevocationSecret(""), /signing secret/);
  assert.equal(executionStatusFromEnv({}, "github-server-secret"), undefined);
  assert.ok(executionStatusFromEnv({ PI_AGENT_REVOCATION_RPC_URL: "http://host.docker.internal:3001" }, "github-server-secret"));
  assert.throws(() => executionStatusFromEnv({ PI_AGENT_REVOCATION_RPC_URL: "http://evil.example:3001" }, "s"), /Invalid/);
});

const mcpSecret = "mcp-test-secret";
const claims: McpClaims = {
  version: 2, run: "exec-1", session: "session-1", slackUser: "U1",
  scope: { repository: "acme/widgets" }, server: "github",
  tools: ["get_file_contents"], mode: "read_only", maxCalls: 12, expiry: Date.now() + 60_000,
};
const initialize = { jsonrpc: "2.0", id: 1, method: "initialize", params: {} };

test("an MCP gateway refuses the token of an execution whose run has ended, and fails closed when it cannot tell", async (t) => {
  let upstreamCalls = 0;
  const upstream = createServer((_request, response) => {
    upstreamCalls++;
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { protocolVersion: "2025-06-18", capabilities: {}, serverInfo: { name: "github", version: "t" } } }));
  });
  const upstreamUrl = await listen(upstream);
  let status: boolean | Error = true;
  const checked: string[] = [];
  const gateway = createMcpGatewayServer({
    server: "github", allowedTools: claims.tools, authorizeTool: authorizeGitHubTool, signingSecret: mcpSecret,
    upstreamUrl: `${upstreamUrl}/mcp`, upstreamHeaders: async () => ({ Authorization: "Bearer upstream" }),
    isActive: async (run: string) => { checked.push(run); if (status instanceof Error) throw status; return status; },
  });
  const url = await listen(gateway);
  t.after(async () => { await close(gateway); await close(upstream); });
  const send = () => fetch(`${url}/mcp`, {
    method: "POST", body: JSON.stringify(initialize),
    headers: { authorization: `Bearer ${issueMcpToken(claims, mcpSecret)}`, "content-type": "application/json", accept: "application/json, text/event-stream" },
  });

  assert.equal((await send()).status, 200);
  assert.equal(upstreamCalls, 1);
  status = false;
  const revoked = await send();
  assert.equal(revoked.status, 401);
  assert.equal((await revoked.json() as { error: { code: string } }).error.code, "token_revoked");
  status = new Error("host unreachable");
  const unknown = await send();
  assert.equal(unknown.status, 503);
  assert.equal((await unknown.json() as { error: { code: string } }).error.code, "revocation_unavailable");
  assert.equal(upstreamCalls, 1);
  assert.deepEqual(checked, ["exec-1", "exec-1", "exec-1"]);
});

test("a classifier token outside a workflow stops working when its run's budget is stopped", async () => {
  let active = true;
  let upstreamCalls = 0;
  const gateway = createModelGateway({
    signingSecret: "signing-secret",
    providerApiKeys: { openrouter: "real-openrouter" },
    budgetLedger: { reserve: () => undefined, settle: () => undefined, firstModel: () => undefined, active: async () => active },
    fetchFn: async () => {
      upstreamCalls++;
      return new Response(JSON.stringify({ model: "typesafe/jev-1.13-20260917", answers: {}, usage: { input_tokens: 1, output_tokens: 1 } }), { status: 200, headers: { "content-type": "application/json" } });
    },
  });
  const base = await listen(gateway);
  const token = issueClassifierGatewayToken({
    kind: "classifier", runId: "solo-run", provider: "openrouter", model: JEV_CLASSIFIER_MODEL.id, api: JEV_CLASSIFIER_MODEL.api,
    maxCalls: 5, expiresAt: Date.now() + 60_000,
  }, "signing-secret");
  const body = { model: JEV_CLASSIFIER_MODEL.id, state: { text: "x" }, questions: { kind: { type: "choice", instructions: "Classify.", criteria: { bug: "b", feature: "f" } } } };
  const ask = () => fetch(`${base}${JEV_CLASSIFIER_MODEL.gatewayPath}`, { method: "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, body: JSON.stringify(body) });
  try {
    assert.equal((await ask()).status, 200);
    active = false;
    const revoked = await ask();
    assert.equal(revoked.status, 429);
    assert.equal((await revoked.json() as { error: { code: string } }).error.code, "token_revoked");
    assert.equal(upstreamCalls, 1);
  } finally { await close(gateway); }
});
