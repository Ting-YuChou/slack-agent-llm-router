import assert from "node:assert/strict";
import { once } from "node:events";
import type { AddressInfo } from "node:net";
import { test } from "node:test";

import { JEV_CLASSIFIER_MODEL, resolveAgentModel } from "../src/agent-model.js";
import { issueClassifierGatewayToken, issueGatewayToken } from "../src/gateway-token.js";
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

test("gateway proxies only token-bound Jev classification and enforces the per-run call budget", async () => {
  const signingSecret = "signing-secret";
  const calls: Array<{ url: string; headers: Headers; body: Record<string, unknown> }> = [];
  const gateway = createModelGateway({
    signingSecret,
    providerApiKeys: { openrouter: "real-openrouter" },
    fetchFn: async (input, init) => {
      calls.push({
        url: String(input),
        headers: new Headers(init?.headers),
        body: JSON.parse(String(init?.body)) as Record<string, unknown>,
      });
      return new Response(JSON.stringify({
        model: "typesafe/jev-1.13-20260917",
        answers: { kind: { type: "choice", choice: "bug", probabilities: { bug: 0.9, feature: 0.1 }, confidence: 0.8 } },
        usage: { input_tokens: 10, output_tokens: 2 },
      }), { status: 200, headers: { "content-type": "application/json" } });
    },
  });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  const token = issueClassifierGatewayToken({
    kind: "classifier",
    runId: "run-jev",
    provider: "openrouter",
    model: JEV_CLASSIFIER_MODEL.id,
    api: JEV_CLASSIFIER_MODEL.api,
    maxCalls: 1,
    expiresAt: Date.now() + 60_000,
  }, signingSecret);
  const body = {
    model: JEV_CLASSIFIER_MODEL.id,
    state: { text: "Login returns 500" },
    questions: {
      kind: {
        type: "choice",
        instructions: "Classify this report.",
        criteria: { bug: "Broken behavior", feature: "New capability" },
      },
    },
  };

  try {
    const first = await fetch(`http://127.0.0.1:${port}${JEV_CLASSIFIER_MODEL.gatewayPath}`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    assert.equal(first.status, 200);
    assert.equal(calls.length, 1);
    assert.equal(calls[0]?.url, JEV_CLASSIFIER_MODEL.upstreamUrl);
    assert.equal(calls[0]?.headers.get("authorization"), "Bearer real-openrouter");
    assert.equal(calls[0]?.headers.get("x-pi-run-id"), "run-jev");
    assert.deepEqual(calls[0]?.body, body);

    const exhausted = await fetch(`http://127.0.0.1:${port}${JEV_CLASSIFIER_MODEL.gatewayPath}`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    assert.equal(exhausted.status, 429);
    assert.equal(calls.length, 1);
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});

test("gateway rejects classifier requests that change the model or question schema", async () => {
  const signingSecret = "signing-secret";
  let upstreamCalls = 0;
  const gateway = createModelGateway({
    signingSecret,
    providerApiKeys: { openrouter: "real-openrouter" },
    fetchFn: async () => {
      upstreamCalls += 1;
      return new Response("{}", { status: 200 });
    },
  });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  const token = issueClassifierGatewayToken({
    kind: "classifier", runId: "run-invalid", provider: "openrouter",
    model: JEV_CLASSIFIER_MODEL.id, api: JEV_CLASSIFIER_MODEL.api,
    maxCalls: 2, expiresAt: Date.now() + 60_000,
  }, signingSecret);

  try {
    for (const body of [
      { model: "~typesafe/jev-latest", state: { text: "x" }, questions: { ok: { type: "noul", instructions: "Is this okay?" } } },
      { model: JEV_CLASSIFIER_MODEL.id, state: { text: "x" }, questions: {} },
      { model: JEV_CLASSIFIER_MODEL.id, state: { text: "x" }, questions: { ok: { type: "choice", instructions: "Pick", criteria: {} } } },
    ]) {
      const response = await fetch(`http://127.0.0.1:${port}${JEV_CLASSIFIER_MODEL.gatewayPath}`, {
        method: "POST",
        headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      assert.equal(response.status, 400);
    }
    assert.equal(upstreamCalls, 0);
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});

test("gateway bounds a stalled Jev classifier request", async () => {
  const signingSecret = "signing-secret";
  const gateway = createModelGateway({
    signingSecret,
    providerApiKeys: { openrouter: "real-openrouter" },
    classifierTimeoutMs: 5,
    fetchFn: async (_input, init) => new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
    }),
  });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  const token = issueClassifierGatewayToken({
    kind: "classifier", runId: "run-timeout", provider: "openrouter",
    model: JEV_CLASSIFIER_MODEL.id, api: JEV_CLASSIFIER_MODEL.api,
    maxCalls: 1, expiresAt: Date.now() + 60_000,
  }, signingSecret);

  try {
    const response = await fetch(`http://127.0.0.1:${port}${JEV_CLASSIFIER_MODEL.gatewayPath}`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({
        model: JEV_CLASSIFIER_MODEL.id,
        state: { text: "Classify this" },
        questions: {
          ok: {
            type: "choice",
            instructions: "Pick one.",
            criteria: { yes: "Yes", no: "No" },
          },
        },
      }),
    });
    assert.equal(response.status, 502);
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});
