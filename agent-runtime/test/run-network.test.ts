import assert from "node:assert/strict";
import { test } from "node:test";

import { buildAgentDockerArgs } from "../src/container-runtime.js";
import { engineVulnerable, parseNetworkAttachments, RunNetworks } from "../src/run-network.js";

test("Docker engines without the CVE-2024-29018 fix are refused", () => {
  for (const version of ["23.0.10", "24.0.9", "25.0.4", "20.10.24"]) assert.equal(engineVulnerable(version), true, version);
  for (const version of ["23.0.11", "25.0.5", "26.0.0", "29.8.2", "27.1.1-rd"]) assert.equal(engineVulnerable(version), false, version);
  assert.equal(engineVulnerable("not-a-version"), true, "an unknown version is treated as unpatched");
});

test("network attachments are container=alias pairs", () => {
  assert.deepEqual(parseNetworkAttachments("slack-pi-model-gateway=model-gateway, slack-pi-mcp-github=mcp-github-gateway"), [
    { container: "slack-pi-model-gateway", alias: "model-gateway" },
    { container: "slack-pi-mcp-github", alias: "mcp-github-gateway" },
  ]);
  assert.deepEqual(parseNetworkAttachments(undefined), []);
  assert.throws(() => parseNetworkAttachments("gateway"), /PI_AGENT_NETWORK_ATTACH/);
  assert.throws(() => parseNetworkAttachments("gate way=alias"), /PI_AGENT_NETWORK_ATTACH/);
});

test("each session gets its own internal network with only the gateways attached, removed on close", async () => {
  const calls: string[][] = [];
  const networks = new RunNetworks({ attachments: [{ container: "gw", alias: "model-gateway" }], docker: async args => { calls.push(args); return ""; } });
  const name = await networks.create("1234abcd-5678-efgh");
  assert.match(name, /^pi-run-[a-z0-9-]+$/);
  assert.deepEqual(calls[0], ["network", "create", "--internal", "--label", "slack-pi-agent-run-network=true", name]);
  assert.deepEqual(calls[1], ["network", "connect", "--alias", "model-gateway", name, "gw"]);
  assert.notEqual(await networks.create("another-session"), name, "sessions never share a network");
  calls.length = 0;
  await networks.remove("1234abcd-5678-efgh");
  assert.deepEqual(calls, [["network", "disconnect", "--force", name, "gw"], ["network", "rm", name]]);
});

test("stale run networks from a previous runtime are removed at boot", async () => {
  const calls: string[][] = [];
  const networks = new RunNetworks({ attachments: [{ container: "gw", alias: "model-gateway" }], docker: async args => {
    calls.push(args);
    return args[1] === "ls" ? "pi-run-old1\npi-run-old2\n" : "";
  } });
  assert.equal(await networks.removeStale(), 2);
  assert.deepEqual(calls[0], ["network", "ls", "--filter", "label=slack-pi-agent-run-network=true", "--format", "{{.Name}}"]);
  assert.ok(calls.some(args => args.join(" ") === "network rm pi-run-old2"));
});

test("agent containers forward no DNS beyond the internal network", () => {
  const args = buildAgentDockerArgs({
    name: "pi-session-1", image: "pi-agent@sha256:abc", network: "pi-run-1", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080/v1", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  });
  const rendered = args.join(" ");
  assert.ok(rendered.includes("--network pi-run-1"));
  assert.ok(rendered.includes("--dns 127.0.0.1"));
});
