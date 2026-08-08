import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_AGENT_MODEL_REF,
  gatewayProviderConfig,
  listAgentModels,
  resolveAgentModel,
} from "../src/agent-model.js";

test("agent model registry exposes one allowlisted model for each supported provider", () => {
  const models = listAgentModels();

  assert.deepEqual(models.map((model) => model.provider), ["openai", "anthropic", "opencode-go"]);
  assert.deepEqual(models.map((model) => model.ref), [
    "openai/gpt-5.6-luna",
    "anthropic/claude-sonnet-4-6",
    "opencode-go/deepseek-v4-pro",
  ]);
  assert.ok(models.every((model) => model.reasoningEffort === "max"));
  assert.ok(models.every((model) => model.gatewayPath.startsWith("/")));
});

test("gateway provider configuration keeps each Pi adapter on its fixed internal path", () => {
  assert.deepEqual(gatewayProviderConfig("http://model-gateway:8080", "openai/gpt-5.6-luna"), {
    provider: "openai",
    api: "openai-responses",
    apiKey: "$OPENAI_API_KEY",
    baseUrl: "http://model-gateway:8080/openai/v1",
  });
  assert.equal(
    gatewayProviderConfig("http://model-gateway:8080", "anthropic/claude-sonnet-4-6").baseUrl,
    "http://model-gateway:8080/anthropic",
  );
  assert.equal(
    gatewayProviderConfig("http://model-gateway:8080", "opencode-go/deepseek-v4-pro").baseUrl,
    "http://model-gateway:8080/opencode-go/v1",
  );
});

test("agent model resolver uses the OpenAI model by default and rejects non-allowlisted models", () => {
  assert.equal(DEFAULT_AGENT_MODEL_REF, "openai/gpt-5.6-luna");
  assert.equal(resolveAgentModel(undefined).ref, DEFAULT_AGENT_MODEL_REF);
  assert.equal(resolveAgentModel("anthropic/claude-sonnet-4-6").credentialEnv, "ANTHROPIC_API_KEY");
  assert.equal(resolveAgentModel("opencode-go/deepseek-v4-pro").api, "openai-completions");
  assert.throws(() => resolveAgentModel("openai/gpt-4o"), /not allowlisted/i);
  assert.throws(() => resolveAgentModel("google/gemini"), /not allowlisted/i);
});
