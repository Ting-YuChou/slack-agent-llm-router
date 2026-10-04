import assert from "node:assert/strict";
import { test } from "node:test";

import { issueMcpToken, verifyMcpToken } from "../src/mcp-token.js";

const claims = {
  run: "run-1",
  session: "session-1",
  slackUser: "U123",
  repository: "Ting-YuChou/slack-agent-llm-router",
  server: "github" as const,
  tools: ["get_file_contents", "search_code"],
  mode: "github_read_only" as const,
  expiry: 2_000,
};

test("MCP token is bound to the run, session, user, repository, server, tools, mode and expiry", () => {
  const token = issueMcpToken(claims, "mcp-secret");
  assert.deepEqual(verifyMcpToken(token, "mcp-secret", 1_000), claims);
  assert.equal(verifyMcpToken(token, "wrong-secret", 1_000), null);
  assert.equal(verifyMcpToken(token, "mcp-secret", 2_001), null);
  assert.equal(verifyMcpToken(`${token}x`, "mcp-secret", 1_000), null);
});

test("MCP token rejects unsupported modes and malformed allowlists", () => {
  const unsupported = issueMcpToken({ ...claims, mode: "approved_write" as any }, "mcp-secret");
  const duplicateTools = issueMcpToken({ ...claims, tools: ["search_code", "search_code"] }, "mcp-secret");
  assert.equal(verifyMcpToken(unsupported, "mcp-secret", 1_000), null);
  assert.equal(verifyMcpToken(duplicateTools, "mcp-secret", 1_000), null);
});
