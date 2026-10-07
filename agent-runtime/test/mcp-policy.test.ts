import assert from "node:assert/strict";
import { test } from "node:test";
import { authorizeClickHouseTool, authorizeContext7Tool, authorizeGitHubTool, sanitizeClickHouseResponse, validateContext7Response } from "../src/mcp-policy.js";
import type { McpClaims } from "../src/mcp-token.js";

const github = { server: "github", scope: { repository: "acme/widgets" } } as unknown as McpClaims;
test("GitHub policy enforces repository, pagination and bounded read methods", () => {
  assert.doesNotThrow(() => authorizeGitHubTool("actions_list", { owner: "acme", repo: "widgets", method: "list_workflow_runs", perPage: 50 }, github));
  assert.throws(() => authorizeGitHubTool("actions_list", { owner: "acme", repo: "widgets", method: "trigger_workflow" }, github), /method/);
  assert.throws(() => authorizeGitHubTool("list_issues", { owner: "acme", repo: "widgets", perPage: 51 }, github), /Page/);
  assert.throws(() => authorizeGitHubTool("get_file_contents", { owner: "other", repo: "widgets" }, github), /Repository/);
});

const clickhouse = { server: "clickhouse", scope: { database: "agent_mcp" } } as unknown as McpClaims;
test("ClickHouse policy only permits bounded SELECTs over aggregate views", () => {
  assert.doesNotThrow(() => authorizeClickHouseTool("run_query", { query: "SELECT model, sum(runs) FROM agent_mcp.model_metrics_daily GROUP BY model" }, clickhouse));
  for (const query of ["SELECT * FROM agent_runs", "DROP TABLE x", "SELECT * FROM url('https://x')", "SELECT 1; SELECT 2"]) {
    assert.throws(() => authorizeClickHouseTool("run_query", { query }, clickhouse), /query/i);
  }
  assert.throws(() => authorizeClickHouseTool("list_tables", { database: "system" }, clickhouse), /database/);
});

test("ClickHouse discovery responses are filtered to the aggregate database", () => {
  const databases = JSON.stringify(["default", "agent_mcp", "system"]);
  const response: any = { result: {
    content: [{ type: "text", text: databases }],
    structuredContent: { result: databases },
  } };
  sanitizeClickHouseResponse("tools/call", "list_databases", response);
  assert.deepEqual(JSON.parse(response.result.content[0].text), ["agent_mcp"]);
  assert.deepEqual(JSON.parse(response.result.structuredContent.result), ["agent_mcp"]);
});

const context7 = { server: "context7", scope: { data: "public_docs" } } as unknown as McpClaims;
test("Context7 policy bounds public documentation inputs", () => {
  assert.doesNotThrow(() => authorizeContext7Tool("query-docs", { libraryId: "/pi/1.0.1", query: "RPC API" }, context7));
  assert.doesNotThrow(() => authorizeContext7Tool("resolve-library-id", { libraryName: "pi", query: "Pi 1.0.1 RPC API" }, context7));
  assert.throws(() => authorizeContext7Tool("resolve-library-id", { libraryName: "x".repeat(201) }, context7), /libraryName/);
  assert.throws(() => authorizeContext7Tool("resolve-library-id", { libraryName: "pi", query: "x".repeat(2001) }, context7), /query/);
  assert.throws(() => authorizeContext7Tool("query-docs", { libraryId: "/x", query: "x".repeat(2001) }, context7), /query/);
});

test("Context7 tools/list fails closed on contract drift", () => {
  assert.doesNotThrow(() => validateContext7Response("tools/list", "tools/list", { result: { tools: [
    { name: "resolve-library-id", inputSchema: { type: "object", properties: { libraryName: {}, query: {} }, required: ["libraryName", "query"] } },
    { name: "query-docs", inputSchema: { type: "object", properties: { libraryId: {}, query: {} }, required: ["libraryId", "query"] } },
  ] } }));
  assert.throws(() => validateContext7Response("tools/list", "tools/list", { result: { tools: [{ name: "renamed", inputSchema: {} }] } }), /contract/);
  assert.throws(() => validateContext7Response("tools/list", "tools/list", { result: { tools: [
    { name: "resolve-library-id", inputSchema: { type: "object", properties: { libraryName: {} }, required: ["libraryName"] } },
    { name: "query-docs", inputSchema: { type: "object", properties: { libraryId: {}, query: {} }, required: ["libraryId", "query"] } },
  ] } }), /contract/);
});
