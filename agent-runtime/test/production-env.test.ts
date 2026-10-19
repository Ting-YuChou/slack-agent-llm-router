import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { createProductionServer } from "../src/server.js";

// Each case fails validation before any Docker call, mkdir or listener; the host coordinator lock lives in a private TMPDIR.
const BASE = { AGENT_RUNTIME_TOKEN: "runtime-token", MODEL_GATEWAY_SIGNING_SECRET: "gateway-secret" };
const KEYS = ["AGENT_RUNTIME_TOKEN", "MODEL_GATEWAY_SIGNING_SECRET", "PI_AGENT_JEV_MODE", "PI_AGENT_JEV_CLASSIFIER_MODE", "OPENROUTER_API_KEY", "PI_AGENT_MCP_MODE", "PI_AGENT_CONFIGURED_PROVIDERS", "PI_AGENT_REPO_PATH", "TMPDIR"];

async function withEnv(env: Record<string, string>, run: () => Promise<void>) {
  const saved = Object.fromEntries(KEYS.map(key => [key, process.env[key]]));
  const tmp = await mkdtemp(path.join(os.tmpdir(), "production-env-"));
  for (const key of KEYS) delete process.env[key];
  Object.assign(process.env, { TMPDIR: tmp, PI_AGENT_REPO_PATH: path.join(tmp, "repo") }, env);
  try { await run(); } finally {
    for (const [key, value] of Object.entries(saved)) { if (value === undefined) delete process.env[key]; else process.env[key] = value; }
    await rm(tmp, { recursive: true, force: true });
  }
}

for (const [name, env, message] of [
  ["a missing runtime token", { MODEL_GATEWAY_SIGNING_SECRET: "gateway-secret" }, /AGENT_RUNTIME_TOKEN and MODEL_GATEWAY_SIGNING_SECRET are required/],
  ["a missing gateway signing secret", { AGENT_RUNTIME_TOKEN: "runtime-token" }, /AGENT_RUNTIME_TOKEN and MODEL_GATEWAY_SIGNING_SECRET are required/],
  ["an empty runtime token", { ...BASE, AGENT_RUNTIME_TOKEN: "" }, /AGENT_RUNTIME_TOKEN and MODEL_GATEWAY_SIGNING_SECRET are required/],
  ["an unknown Jev mode", { ...BASE, PI_AGENT_JEV_MODE: "always" }, /PI_AGENT_JEV_MODE must be off, shadow, or on/],
  ["an unknown Jev classifier mode", { ...BASE, PI_AGENT_JEV_CLASSIFIER_MODE: "maybe" }, /PI_AGENT_JEV_CLASSIFIER_MODE must be off or on/],
  ["Jev classification without an OpenRouter key", { ...BASE, PI_AGENT_JEV_CLASSIFIER_MODE: "on" }, /OPENROUTER_API_KEY is required/],
  ["the unreleased approved_write MCP mode", { ...BASE, PI_AGENT_MCP_MODE: "approved_write" }, /approved_write is not enabled/],
  ["an unknown MCP mode", { ...BASE, PI_AGENT_MCP_MODE: "write" }, /PI_AGENT_MCP_MODE must be off, read_only, or github_read_only/],
  ["no configured model provider", { ...BASE, PI_AGENT_CONFIGURED_PROVIDERS: "nobody" }, /At least one Agent model provider must be configured/],
] as const) {
  test(`createProductionServer refuses to start with ${name}`, async () => {
    await withEnv(env, async () => {
      await assert.rejects(createProductionServer(), message);
      // The failed start released the host coordinator, so a corrected restart is not locked out.
      await assert.rejects(createProductionServer(), message);
    });
  });
}
