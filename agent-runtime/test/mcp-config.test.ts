import assert from "node:assert/strict";
import { test } from "node:test";

import { buildMcpRegistration, parseMcpMode } from "../src/mcp-config.js";

test("MCP mode defaults to off and reserves approved_write for a later release", () => {
  assert.equal(parseMcpMode(undefined), "off");
  assert.equal(parseMcpMode("off"), "off");
  assert.equal(parseMcpMode("github_read_only"), "github_read_only");
  assert.throws(() => parseMcpMode("approved_write"), /not enabled/);
  assert.throws(() => parseMcpMode("unknown"), /PI_AGENT_MCP_MODE/);
});

test("trusted MCP registration exposes only explicitly allowlisted direct tools", () => {
  assert.deepEqual(buildMcpRegistration({
    mode: "github_read_only",
    gatewayUrl: "http://mcp-gateway:8090/mcp",
    token: "short-lived-token",
    tools: ["get_file_contents", "search_code"],
  }), {
    url: "http://mcp-gateway:8090/mcp",
    headers: { Authorization: "Bearer short-lived-token" },
    exposure: "hidden",
    toolExposure: {
      get_file_contents: "direct",
      search_code: "direct",
    },
    timeout: 30_000,
  });
});

test("trusted MCP registration rejects external and credential-bearing URLs", () => {
  for (const url of [
    "https://github.com/mcp",
    "http://127.0.0.1:8090/mcp",
    "http://user:pass@mcp-gateway:8090/mcp",
    "http://mcp-gateway:8090/other",
  ]) {
    assert.throws(() => buildMcpRegistration({
      mode: "github_read_only", gatewayUrl: url, token: "token", tools: ["search_code"],
    }), /gateway/);
  }
});
