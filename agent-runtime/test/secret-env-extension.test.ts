import assert from "node:assert/strict";
import { test } from "node:test";
import path from "node:path";
import { pathToFileURL } from "node:url";

const extensionUrl = pathToFileURL(path.resolve(import.meta.dirname, "../../extensions/secret-env.ts")).href;

test("bash children never inherit gateway, provider or MCP credentials", async () => {
  const { scrubSecretEnv } = await import(extensionUrl);
  const scrubbed = scrubSecretEnv({
    PATH: "/usr/bin", HOME: "/tmp/pi-home", PI_AGENT_WORKTREE: "/w", PI_AGENT_SAFE_COMMANDS_JSON: "[]",
    OPENAI_API_KEY: "t", ANTHROPIC_API_KEY: "t", OPENCODE_API_KEY: "t", OPENROUTER_API_KEY: "t",
    PI_AGENT_MCP_SERVERS_JSON: "[{\"token\":\"t\"}]", GITHUB_TOKEN: "t", AWS_SECRET_ACCESS_KEY: "t", SOME_PASSWORD: "t",
  });
  assert.deepEqual(Object.keys(scrubbed).sort(), ["HOME", "PATH", "PI_AGENT_SAFE_COMMANDS_JSON", "PI_AGENT_WORKTREE"]);
});

test("the extension replaces the built-in bash tool with one that scrubs its environment", async () => {
  const { default: extension } = await import(`${extensionUrl}?${Math.random()}`);
  const tools: Array<{ name: string; execute: unknown }> = [];
  extension({ registerTool: (tool: { name: string; execute: unknown }) => tools.push(tool), on: () => undefined });
  assert.deepEqual(tools.map((tool) => tool.name), ["bash"]);
  assert.equal(typeof tools[0].execute, "function");
});
