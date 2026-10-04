import assert from "node:assert/strict";
import { once } from "node:events";
import type { AddressInfo } from "node:net";
import { test } from "node:test";

import { resolveAgentModel } from "../src/agent-model.js";
import { issueGatewayToken } from "../src/gateway-token.js";
import { createModelGateway, validateAgentModelRequest } from "../src/model-gateway.js";

test("gateway validates each request against the token-bound model and reasoning effort", () => {
  const openai = resolveAgentModel("openai/gpt-5.6-luna");
  const anthropic = resolveAgentModel("anthropic/claude-sonnet-4-6");
  const opencode = resolveAgentModel("opencode-go/deepseek-v4-pro");

  assert.equal(validateAgentModelRequest({ model: openai.id, reasoning: { effort: "max" } }, openai), true);
  assert.equal(validateAgentModelRequest({ model: openai.id, reasoning: { effort: "high" } }, openai), false);
  const sol = resolveAgentModel("openai/gpt-5.6-sol");
  assert.equal(validateAgentModelRequest({ model: sol.id, reasoning: { effort: "high" } }, sol, "high"), true);
  assert.equal(validateAgentModelRequest({ model: openai.id, reasoning: { effort: "high" } }, sol, "high"), false);
  assert.equal(validateAgentModelRequest({ model: sol.id, reasoning: { effort: "max" } }, sol, "high"), false);
  assert.equal(validateAgentModelRequest({ model: anthropic.id, output_config: { effort: "max" } }, anthropic), true);
  assert.equal(validateAgentModelRequest({ model: anthropic.id, output_config: { effort: "high" } }, anthropic), false);
  assert.equal(validateAgentModelRequest({
    model: opencode.id,
    thinking: { type: "enabled" },
    reasoning_effort: "max",
  }, opencode), true);
  assert.equal(validateAgentModelRequest({ model: opencode.id, thinking: { type: "enabled" } }, opencode), false);
  assert.equal(validateAgentModelRequest({
    model: opencode.id,
    thinking: { type: "enabled" },
    reasoning_effort: "high",
  }, opencode), false);
  assert.equal(validateAgentModelRequest({ model: opencode.id, thinking: { type: "disabled" } }, opencode), false);
  assert.equal(validateAgentModelRequest({ model: "other" }, anthropic), false);
});

test("gateway routes only the token-bound provider path and injects that provider's real key", async () => {
  const signingSecret = "signing-secret";
  const calls: Array<{ url: string; headers: Headers }> = [];
  const gateway = createModelGateway({
    signingSecret,
    providerApiKeys: {
      openai: "real-openai",
      anthropic: "real-anthropic",
      "opencode-go": "real-opencode",
    },
    fetchFn: async (input, init) => {
      calls.push({ url: String(input), headers: new Headers(init?.headers) });
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    },
  });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  const model = resolveAgentModel("anthropic/claude-sonnet-4-6");
  const token = issueGatewayToken({
    runId: "run-anthropic",
    provider: model.provider,
    model: model.id,
    api: model.api,
    reasoningEffort: model.reasoningEffort,
    expiresAt: Date.now() + 60_000,
  }, signingSecret);

  try {
    const response = await fetch(`http://127.0.0.1:${port}${model.gatewayPath}`, {
      method: "POST",
      headers: { "x-api-key": token, "content-type": "application/json", "anthropic-version": "2023-06-01" },
      body: JSON.stringify({ model: model.id, messages: [], output_config: { effort: "max" } }),
    });
    assert.equal(response.status, 200);
    assert.equal(calls.length, 1);
    assert.equal(calls[0]?.url, model.upstreamUrl);
    assert.equal(calls[0]?.headers.get("x-api-key"), "real-anthropic");
    assert.equal(calls[0]?.headers.get("authorization"), null);
    assert.equal(calls[0]?.headers.get("x-pi-run-id"), "run-anthropic");

    const wrongPath = await fetch(`http://127.0.0.1:${port}/openai/v1/responses`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ model: model.id }),
    });
    assert.equal(wrongPath.status, 403);
    assert.equal(calls.length, 1);
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});

test("gateway accepts token-bound requests for both OpenAI models", async () => {
  const signingSecret = "signing-secret";
  const forwarded: Array<{ model: string; effort: string }> = [];
  const gateway = createModelGateway({
    signingSecret,
    providerApiKeys: { openai: "real-openai" },
    fetchFn: async (_input, init) => {
      const body = JSON.parse(String(init?.body));
      forwarded.push({ model: body.model, effort: body.reasoning.effort });
      return new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "content-type": "application/json" } });
    },
  });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  const cases = [
    { ref: "openai/gpt-5.6-luna", effort: "max" as const },
    { ref: "openai/gpt-5.6-sol", effort: "high" as const },
  ];

  try {
    for (const item of cases) {
      const model = resolveAgentModel(item.ref);
      const token = issueGatewayToken({
        runId: `run-${model.id}`,
        provider: model.provider,
        model: model.id,
        api: model.api,
        reasoningEffort: item.effort,
        expiresAt: Date.now() + 60_000,
      }, signingSecret);
      const response = await fetch(`http://127.0.0.1:${port}${model.gatewayPath}`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
        body: JSON.stringify({ model: model.id, reasoning: { effort: item.effort }, input: "test" }),
      });
      assert.equal(response.status, 200);
    }
    assert.deepEqual(forwarded, [
      { model: "gpt-5.6-luna", effort: "max" },
      { model: "gpt-5.6-sol", effort: "high" },
    ]);
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});
