import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildClassifierGatewayProviderRegistration,
  buildGatewayProviderRegistrations,
  DEFAULT_AGENT_MODEL_REF,
  gatewayProviderConfig,
  listAgentModels,
  resolveAgentModel,
} from "../src/agent-model.js";

test("Pi provider registrations contain both OpenAI models in one provider entry", () => {
  const registrations = buildGatewayProviderRegistrations("http://model-gateway:8080");
  const openai = registrations.filter((registration) => registration.provider === "openai");
  assert.equal(openai.length, 1);
  assert.deepEqual(openai[0]?.models.map((model) => model.id), ["gpt-5.6-luna", "gpt-5.6-sol"]);
});

test("agent model registry exposes Luna and Sol through the OpenAI provider", () => {
  const models = listAgentModels();

  assert.deepEqual(models.map((model) => model.provider), ["openai", "openai", "anthropic", "opencode-go"]);
  assert.deepEqual(models.map((model) => model.ref), [
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-sol",
    "anthropic/claude-sonnet-4-6",
    "opencode-go/deepseek-v4-pro",
  ]);
  assert.ok(models.every((model) => model.reasoningEffort === "max"));
  assert.ok(models.every((model) => model.gatewayPath.startsWith("/")));
  assert.equal(resolveAgentModel("openai/gpt-5.6-sol").api, "openai-responses");
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

test("Jev classifier registration preserves Pi's built-in OpenRouter catalog behind the internal gateway", () => {
  assert.deepEqual(buildClassifierGatewayProviderRegistration("http://model-gateway:8080"), {
    provider: "openrouter",
    apiKey: "$OPENROUTER_API_KEY",
    baseUrl: "http://model-gateway:8080/openrouter/api/v1",
  });
});

test("agent model resolver uses the OpenAI model by default and rejects non-allowlisted models", () => {
  assert.equal(DEFAULT_AGENT_MODEL_REF, "openai/gpt-5.6-luna");
  assert.equal(resolveAgentModel(undefined).ref, DEFAULT_AGENT_MODEL_REF);
  assert.equal(resolveAgentModel("anthropic/claude-sonnet-4-6").credentialEnv, "ANTHROPIC_API_KEY");
  assert.equal(resolveAgentModel("opencode-go/deepseek-v4-pro").api, "openai-completions");
  assert.throws(() => resolveAgentModel("openai/gpt-4o"), /not allowlisted/i);
  assert.throws(() => resolveAgentModel("google/gemini"), /not allowlisted/i);
});
