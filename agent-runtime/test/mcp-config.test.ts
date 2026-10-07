import assert from "node:assert/strict";
import { test } from "node:test";

import { buildMcpRegistrations, codemodeMcpPrompt, enabledMcpServers, namespacedMcpToolNames, parseMcpMode, parseMcpServers, MCP_SERVERS } from "../src/mcp-config.js";

test("MCP modes keep legacy GitHub compatibility and default to off", () => {
  assert.equal(parseMcpMode(undefined), "off");
  assert.equal(parseMcpMode("read_only"), "read_only");
  assert.equal(parseMcpMode("github_read_only"), "github_read_only");
  assert.throws(() => parseMcpMode("approved_write"), /not enabled/);
});

test("codemode instructions bound summaries and treat MCP output as data", () => {
  const prompt = codemodeMcpPrompt(["github", "clickhouse"]);
  assert.match(prompt, /searchTools\(\)/);
  assert.match(prompt, /8 KiB/);
  assert.match(prompt, /untrusted data/);
});

test("server list is fixed, deduplicated, and ignored while off", () => {
  assert.deepEqual(parseMcpServers("github,clickhouse,github,context7,codegraph"), ["github", "clickhouse", "context7", "codegraph"]);
  assert.throws(() => parseMcpServers("github,evil"), /PI_AGENT_MCP_SERVERS/);
  assert.deepEqual(enabledMcpServers("off", ["github"]), []);
  assert.deepEqual(enabledMcpServers("github_read_only", ["clickhouse"]), ["github"]);
});

test("trusted registrations preserve direct and codemode exposure per server", () => {
  const registrations = buildMcpRegistrations([
    { server: "github", token: "gh-token", gatewayUrl: "http://mcp-github:8090/mcp" },
    { server: "context7", token: "ctx-token", gatewayUrl: "http://mcp-context7:8090/mcp" },
    { server: "codegraph", command: "/usr/local/bin/codegraph-lazy-proxy" },
  ]);
  assert.equal(registrations.github.toolExposure.get_file_contents, "direct");
  assert.equal(registrations.github.toolExposure.actions_list, "codemode");
  assert.equal(registrations.context7.toolExposure["query-docs"], "codemode");
  assert.equal("command" in registrations.codegraph ? registrations.codegraph.command : "", "/usr/local/bin/codegraph-lazy-proxy");
  assert.equal(registrations.codegraph.exposure, "hidden");
});

test("registrations reject untrusted URLs and missing credentials", () => {
  assert.throws(() => buildMcpRegistrations([{ server: "github", token: "x", gatewayUrl: "https://example.com/mcp" }]), /gateway/);
  assert.throws(() => buildMcpRegistrations([{ server: "clickhouse", token: "", gatewayUrl: "http://mcp-clickhouse-gateway:8090/mcp" }]), /token/);
});

test("only direct tools are added to Pi's global tool list", () => {
  assert.deepEqual(namespacedMcpToolNames("github", MCP_SERVERS.github.directTools), [
    "mcp__github__get_file_contents", "mcp__github__search_code", "mcp__github__issue_read", "mcp__github__pull_request_read",
  ]);
});
