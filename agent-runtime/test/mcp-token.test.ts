import assert from "node:assert/strict";
import { test } from "node:test";

import { deriveMcpServerSecret, issueMcpToken, verifyMcpToken } from "../src/mcp-token.js";

const claims = {
  version: 2 as const, run: "run-1", session: "session-1", slackUser: "U123",
  server: "github" as const, mode: "read_only" as const,
  scope: { repository: "acme/widgets" }, tools: ["get_file_contents"], maxCalls: 12, expiry: 2_000,
};

test("v2 MCP tokens bind server, scope, tools, call budget and expiry", () => {
  const secret = deriveMcpServerSecret("root-secret", "github");
  const token = issueMcpToken(claims, secret);
  assert.deepEqual(verifyMcpToken(token, secret, "github", 1_000), claims);
  assert.equal(verifyMcpToken(token, secret, "clickhouse", 1_000), null);
  assert.equal(verifyMcpToken(token, secret, "github", 2_001), null);
});

test("per-server derived keys cannot verify another server token", () => {
  const github = deriveMcpServerSecret("root-secret", "github");
  const context7 = deriveMcpServerSecret("root-secret", "context7");
  const token = issueMcpToken(claims, github);
  assert.equal(verifyMcpToken(token, context7, "github", 1_000), null);
});

test("invalid scopes, duplicate tools, and nonpositive budgets are rejected", () => {
  const secret = deriveMcpServerSecret("root-secret", "github");
  for (const invalid of [
    { ...claims, maxCalls: 0 },
    { ...claims, tools: ["get_file_contents", "get_file_contents"] },
    { ...claims, scope: { repository: "bad" } },
    { ...claims, mode: "write" },
  ]) assert.equal(verifyMcpToken(issueMcpToken(invalid as any, secret), secret, "github", 1_000), null);
});
