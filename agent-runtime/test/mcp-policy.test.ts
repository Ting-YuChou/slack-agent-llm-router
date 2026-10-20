import assert from "node:assert/strict";
import { test } from "node:test";
import { authorizeClickHouseTool, authorizeContext7Tool, authorizeGitHubTool, githubMcpTokenScope, sanitizeClickHouseResponse, validateContext7Response } from "../src/mcp-policy.js";
import type { McpClaims } from "../src/mcp-token.js";

const github = { server: "github", scope: { repository: "acme/widgets" } } as unknown as McpClaims;
test("GitHub policy enforces repository, pagination and bounded read methods", () => {
  assert.doesNotThrow(() => authorizeGitHubTool("actions_list", { owner: "acme", repo: "widgets", method: "list_workflow_runs", perPage: 50 }, github));
  assert.throws(() => authorizeGitHubTool("actions_list", { owner: "acme", repo: "widgets", method: "trigger_workflow" }, github), /method/);
  assert.throws(() => authorizeGitHubTool("list_issues", { owner: "acme", repo: "widgets", perPage: 51 }, github), /Page/);
  assert.throws(() => authorizeGitHubTool("get_file_contents", { owner: "other", repo: "widgets" }, github), /Repository/);
});

test("GitHub search_code stays pinned to the scoped repository", () => {
  for (const query of [
    "repo:acme/widgets password",
    "Repo:ACME/Widgets language:typescript path:src/ token",
    'repo:acme/widgets "foo OR (bar) org:x"',
    "repo:acme/widgets NOT test",
    "repo:acme/widgets myrepo:other",
  ]) assert.doesNotThrow(() => authorizeGitHubTool("search_code", { query }, github), query);
  for (const query of [
    "password",
    "repo:acme/secret password",
    "repo:acme/widgets OR (repo:acme/secret password)",
    "repo:acme/widgets OR repo:acme/secret",
    "repo:acme/widgets or repo:acme/secret",
    "(repo:acme/widgets) password",
    "repo:acme/widgets repo:acme/secret",
    "repo:acme/widgets -repo:acme/widgets",
    "repo:acme/widgets NOT repo:acme/other",
    "repo:acme/widgets org:acme password",
    "repo:acme/widgets user:someone",
    "repo:acme/widgets owner:acme",
    "repo:acme/widgets enterprise:corp",
    "repo:acme/widgets x,org:acme",
    'repo:"acme/secret"',
    'repo:acme/widgets "unbalanced',
    'repo:acme/widgets "a\\" repo:acme/secret "',
  ]) assert.throws(() => authorizeGitHubTool("search_code", { query }, github), /Repository/, query);
  assert.throws(() => authorizeGitHubTool("search_code", { query: 42 }, github), /Repository/);
});

test("GitHub MCP installation token is scoped to configured repositories with read-only permissions", () => {
  const scope = githubMcpTokenScope("acme/widgets,acme/gadgets");
  assert.deepEqual(scope.repositories, ["widgets", "gadgets"]);
  assert.ok(Object.values(scope.permissions).every((level) => level === "read"));
  assert.equal(scope.permissions.contents, "read");
  assert.throws(() => githubMcpTokenScope(undefined), /PI_AGENT_GITHUB_REPOSITORIES/);
  assert.throws(() => githubMcpTokenScope("acme/widgets,other/gadgets"), /single installation owner/);
});

const clickhouse = { server: "clickhouse", scope: { database: "agent_mcp" } } as unknown as McpClaims;
test("ClickHouse policy only permits bounded SELECTs over aggregate views", () => {
  assert.doesNotThrow(() => authorizeClickHouseTool("run_query", { query: "SELECT model, sum(runs) FROM agent_mcp.model_metrics_daily GROUP BY model" }, clickhouse));
  for (const query of ["SELECT * FROM agent_runs", "DROP TABLE x", "SELECT * FROM url('https://x')", "SELECT 1; SELECT 2"]) {
    assert.throws(() => authorizeClickHouseTool("run_query", { query }, clickhouse), /query/i);
  }
  assert.throws(() => authorizeClickHouseTool("list_tables", { database: "system" }, clickhouse), /database/);
});

test("ClickHouse policy keeps allowed aggregate queries working", () => {
  for (const query of [
    "SELECT model, sum(runs) FROM agent_mcp.model_metrics_daily GROUP BY model ORDER BY model, day LIMIT 10",
    "select hour, errors from run_metrics_hourly where status = 'failed' and model in ('a', 'b''c')",
    "SELECT r.hour, t.calls FROM agent_mcp.run_metrics_hourly AS r JOIN agent_mcp.tool_metrics_hourly AS t ON r.hour = t.hour",
    "SELECT count() FROM (SELECT hour, sum(spans) FROM agent_mcp.trace_metrics_hourly GROUP BY hour) AS s",
    "SELECT * FROM agent_mcp.test_metrics_hourly WHERE hour IN (SELECT hour FROM agent_mcp.run_metrics_hourly)",
    "SELECT * FROM agent_mcp.run_metrics_hourly UNION ALL SELECT * FROM agent_mcp.run_metrics_hourly;",
  ]) {
    const args: Record<string, unknown> = { query };
    assert.doesNotThrow(() => authorizeClickHouseTool("run_query", args, clickhouse), query);
    assert.match(String(args.query), /\nSETTINGS max_execution_time=5, max_result_rows=1000, result_overflow_mode='break', max_memory_usage=268435456$/);
    assert.doesNotMatch(String(args.query), /;/);
  }
});

test("ClickHouse policy denies table, comment, quoting and table-function bypasses", () => {
  for (const query of [
    "SELECT * FROM agent_mcp.run_metrics_hourly, default.agent_runs",
    "SELECT * FROM agent_mcp.run_metrics_hourly AS r, default.agent_runs AS a",
    "SELECT * FROM agent_mcp.run_metrics_hourly r ,default.agent_runs",
    "SELECT * FROM agent_mcp.run_metrics_hourly AS where, default.agent_runs",
    "SELECT * FROM agent_mcp.run_metrics_hourly JOIN agent_mcp.tool_metrics_hourly ON 1 = 1, default.agent_runs",
    "SELECT * FROM (SELECT 1 FROM agent_mcp.run_metrics_hourly) AS s, default.agent_runs",
    "SELECT * FROM `default`.`agent_runs`",
    "SELECT * FROM `agent_mcp.run_metrics_hourly`",
    'SELECT * FROM "default"."agent_runs"',
    'SELECT * FROM agent_mcp."run_metrics_hourly"',
    "SELECT * FROM/**/default.agent_runs",
    "SELECT * FROM agent_mcp.run_metrics_hourly /* note */",
    "SELECT * FROM agent_mcp.run_metrics_hourly --",
    "SELECT * FROM agent_mcp.run_metrics_hourly -- trailing comment",
    "SELECT * FROM agent_mcp.run_metrics_hourly # comment",
    "SELECT * FROM remoteSecure('host', default.agent_runs)",
    "SELECT * FROM mysql('host:3306', 'db', 'table', 'user', 'password')",
    "SELECT * FROM postgresql('host:5432', 'db', 'table', 'user', 'password')",
    "SELECT * FROM s3Cluster('cluster', 'https://bucket/key')",
    "SELECT * FROM numbers(10)",
    "SELECT * FROM system.numbers (10)",
    "SELECT * FROM agent_mcp.run_metrics_hourly(1)",
    "SELECT * FROM agent_mcp.run_metrics_hourly JOIN generateRandom('a UInt8') AS g ON 1 = 1",
    "SELECT * FROM agent_mcp.run_metrics_hourly WHERE hour IN (SELECT hour FROM default.agent_runs)",
    "SELECT * FROM agent_mcp.run_metrics_hourly WHERE run_id IN default.agent_runs",
    "SELECT * FROM (default.agent_runs)",
    "SELECT * FROM agent_mcp.run_metrics_hourly WHERE 1 IN remoteSecure('host', default.agent_runs)",
    "SELECT 1from default.agent_runs",
    "SELECT * FROM agent_mcp.run_metrics_hourly INTO OUTFILE 'x.csv'",
    "SELECT * FROM default.agent_mcp.run_metrics_hourly",
    "SELECT * FROM agent_mcp.run_metrics_hourly WHERE model = 'unterminated",
    "SELECT * FROM agent_mcp.run_metrics_hourly WHERE model = 'a\\' , default.agent_runs'",
    "SELECT * FROM agent_mcp.run_metrics_hourly WHERE model = $$x$$",
    "SELECT * FROM default.agent_runs",
    "SELECT * FROM agent_mcp.run_metrics_hourly; SELECT * FROM default.agent_runs",
    "SELECT * FROM agent_mcp.run_metrics_hourly;;",
    "SELECT 1",
  ]) {
    assert.throws(() => authorizeClickHouseTool("run_query", { query }, clickhouse), (error: any) => error.code === "policy_denied" && /query/i.test(error.message), query);
  }
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
