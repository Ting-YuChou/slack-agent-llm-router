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

test("gateway rejects hosted tools, strips unknown betas and accepts Anthropic SDK query strings", async () => {
  const signingSecret = "signing-secret";
  const calls: Array<{ url: string; headers: Headers; body: any }> = [];
  const gateway = createModelGateway({
    signingSecret, providerApiKeys: { openai: "real-openai", anthropic: "real-anthropic" },
    fetchFn: async (input, init) => {
      calls.push({ url: String(input), headers: new Headers(init?.headers), body: JSON.parse(String(init?.body)) });
      return new Response("{}", { status: 200, headers: { "content-type": "application/json" } });
    },
  });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  const tokenFor = (ref: string) => { const model = resolveAgentModel(ref); return issueGatewayToken({ runId: "r", provider: model.provider, model: model.id, api: model.api, reasoningEffort: model.reasoningEffort, expiresAt: Date.now() + 60_000 }, signingSecret); };
  try {
    const luna = resolveAgentModel("openai/gpt-5.6-luna");
    const hosted = await fetch(`http://127.0.0.1:${port}${luna.gatewayPath}`, {
      method: "POST", headers: { authorization: `Bearer ${tokenFor(luna.ref)}`, "content-type": "application/json" },
      body: JSON.stringify({ model: luna.id, reasoning: { effort: "max" }, input: "x", tools: [{ type: "mcp", server_url: "https://evil.example" }] }),
    });
    assert.equal(hosted.status, 400);
    assert.deepEqual(await hosted.json(), { error: { code: "request_field_not_allowed", message: "Request field is not allowed through the gateway: tools[0].type" } });
    assert.equal(calls.length, 0);

    const sonnet = resolveAgentModel("anthropic/claude-sonnet-4-6");
    const ok = await fetch(`http://127.0.0.1:${port}${sonnet.gatewayPath}?beta=true`, {
      method: "POST", headers: { "x-api-key": tokenFor(sonnet.ref), "content-type": "application/json", "anthropic-beta": "mcp-client-2025-04-04,interleaved-thinking-2025-05-14" },
      body: JSON.stringify({ model: sonnet.id, messages: [], output_config: { effort: "max" } }),
    });
    assert.equal(ok.status, 200);
    assert.equal(calls[0]?.url, sonnet.upstreamUrl);
    assert.equal(calls[0]?.headers.get("anthropic-beta"), "interleaved-thinking-2025-05-14");

    const responses = await fetch(`http://127.0.0.1:${port}${luna.gatewayPath}`, {
      method: "POST", headers: { authorization: `Bearer ${tokenFor(luna.ref)}`, "content-type": "application/json" },
      body: JSON.stringify({ model: luna.id, reasoning: { effort: "max" }, input: "x" }),
    });
    assert.equal(responses.status, 200);
    assert.equal(calls[1]?.body.store, false);

    const wrong = await fetch(`http://127.0.0.1:${port}/anthropic/v1/other?beta=true`, {
      method: "POST", headers: { "x-api-key": tokenFor(sonnet.ref), "content-type": "application/json" }, body: "{}",
    });
    assert.equal(wrong.status, 403);
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});

test("with a budget store, tokens without a budget identity are refused and role caps reach the provider", async () => {
  const signingSecret = "signing-secret";
  const bodies: any[] = [];
  const reserved: string[] = [];
  const gateway = createModelGateway({
    signingSecret, providerApiKeys: { openai: "real-openai" }, prices: { "openai/gpt-5.6-luna": { input: 1, output: 1 } },
    budgetLedger: { reserve: async (workflow: string) => { reserved.push(workflow); }, settle: async () => {}, firstModel: async () => undefined } as any,
    fetchFn: async (_input, init) => { bodies.push(JSON.parse(String(init?.body))); return new Response("{}", { status: 200 }); },
  });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  const luna = resolveAgentModel("openai/gpt-5.6-luna");
  const send = (claims: Record<string, unknown>) => fetch(`http://127.0.0.1:${port}${luna.gatewayPath}`, {
    method: "POST", headers: { authorization: `Bearer ${issueGatewayToken({ runId: "r", provider: luna.provider, model: luna.id, api: luna.api, reasoningEffort: "max", expiresAt: Date.now() + 60_000, ...claims } as any, signingSecret)}`, "content-type": "application/json" },
    body: JSON.stringify({ model: luna.id, reasoning: { effort: "max" }, input: "x", max_output_tokens: 128000 }),
  });
  try {
    const unbudgeted = await send({});
    assert.equal(unbudgeted.status, 429);
    assert.equal((await unbudgeted.json()).error.code, "workflow_budget_unavailable_or_exhausted");
    assert.equal(bodies.length, 0);
    const writer = await send({ workflowRunId: "run-budget", role: "writer", outputTokenLimit: 32768 });
    assert.equal(writer.status, 200);
    assert.equal(bodies[0].max_output_tokens, 32768);
    assert.deepEqual(reserved, ["run-budget"]);
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});

test("a single run token cannot open more than four concurrent provider streams", { timeout: 5_000 }, async () => {
  const signingSecret = "signing-secret";
  let release!: () => void;
  const hold = new Promise<void>((resolve) => { release = resolve; });
  const gateway = createModelGateway({
    signingSecret, providerApiKeys: { openai: "real-openai" },
    fetchFn: async () => { await hold; return new Response("{}", { status: 200 }); },
  });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  const luna = resolveAgentModel("openai/gpt-5.6-luna");
  const token = issueGatewayToken({ runId: "busy", provider: luna.provider, model: luna.id, api: luna.api, reasoningEffort: "max", expiresAt: Date.now() + 60_000 }, signingSecret);
  const send = () => fetch(`http://127.0.0.1:${port}${luna.gatewayPath}`, { method: "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, body: JSON.stringify({ model: luna.id, reasoning: { effort: "max" }, input: "x" }) });
  try {
    const first = Array.from({ length: 4 }, send);
    await new Promise((resolve) => setTimeout(resolve, 50));
    const fifth = await send();
    assert.equal(fifth.status, 429);
    assert.equal((await fifth.json()).error.code, "too_many_streams");
    release();
    assert.deepEqual((await Promise.all(first)).map((r) => r.status), [200, 200, 200, 200]);
    assert.equal((await send()).status, 200);
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});

test("the stream cap holds while budget admission is pending and slots free on every failure path", { timeout: 5_000 }, async () => {
  const signingSecret = "signing-secret";
  let open!: () => void;
  let gate = new Promise<void>((resolve) => { open = resolve; });
  let refuse = false;
  let upstreamFails = false;
  const gateway = createModelGateway({
    signingSecret, providerApiKeys: { openai: "real-openai" }, prices: { "openai/gpt-5.6-luna": { input: 1, output: 1 } },
    budgetLedger: {
      reserve: async () => { await gate; if (refuse) throw new Error("exhausted"); },
      settle: async () => {}, firstModel: async () => undefined,
    },
    fetchFn: async () => { if (upstreamFails) throw new Error("down"); return new Response("{}", { status: 200 }); },
  });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  const luna = resolveAgentModel("openai/gpt-5.6-luna");
  const token = issueGatewayToken({ runId: "racy", workflowRunId: "wf", provider: luna.provider, model: luna.id, api: luna.api, reasoningEffort: "max", expiresAt: Date.now() + 60_000 }, signingSecret);
  const send = () => fetch(`http://127.0.0.1:${port}${luna.gatewayPath}`, { method: "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, body: JSON.stringify({ model: luna.id, reasoning: { effort: "max" }, input: "x" }) });
  const codes = async (responses: Response[]) => Promise.all(responses.map(async (r) => r.status === 200 ? 200 : `${r.status} ${(await r.json()).error.code}`));
  try {
    const racing = Array.from({ length: 6 }, send);
    await new Promise((resolve) => setTimeout(resolve, 100));
    open();
    const statuses = await codes(await Promise.all(racing));
    assert.equal(statuses.filter((s) => s === "429 too_many_streams").length, 2, JSON.stringify(statuses));
    assert.equal(statuses.filter((s) => s === 200).length, 4, JSON.stringify(statuses));

    refuse = true;
    assert.deepEqual(await codes(await Promise.all(Array.from({ length: 4 }, send))), Array(4).fill("429 workflow_budget_unavailable_or_exhausted"));
    refuse = false;
    upstreamFails = true;
    assert.deepEqual(await codes(await Promise.all(Array.from({ length: 4 }, send))), Array(4).fill("502 upstream_unavailable"));
    upstreamFails = false;
    gate = new Promise<void>((resolve) => { open = resolve; });
    const afterFailures = Array.from({ length: 4 }, send);
    await new Promise((resolve) => setTimeout(resolve, 100));
    open();
    assert.deepEqual(await codes(await Promise.all(afterFailures)), [200, 200, 200, 200]);
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});

test("the gateway process refuses to start without a budget store", async () => {
  const { gatewayLedgerFromEnv } = await import("../src/model-gateway.js");
  assert.throws(() => gatewayLedgerFromEnv({}, "secret"), /PI_AGENT_BUDGET_RPC_URL/);
  assert.ok(gatewayLedgerFromEnv({ PI_AGENT_BUDGET_RPC_URL: "http://host.docker.internal:3001" }, "secret"));
});

test("a run token is capped on total model requests, not only concurrent ones", { timeout: 5_000 }, async () => {
  const signingSecret = "signing-secret";
  const gateway = createModelGateway({ signingSecret, providerApiKeys: { openai: "real-openai" }, maxRequestsPerToken: 3,
    fetchFn: async () => new Response("{}", { status: 200 }) });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  const luna = resolveAgentModel("openai/gpt-5.6-luna");
  const tokenFor = (runId: string) => issueGatewayToken({ runId, provider: luna.provider, model: luna.id, api: luna.api, reasoningEffort: "max", expiresAt: Date.now() + 60_000 }, signingSecret);
  const send = (token: string) => fetch(`http://127.0.0.1:${port}${luna.gatewayPath}`, { method: "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, body: JSON.stringify({ model: luna.id, reasoning: { effort: "max" }, input: "x" }) });
  try {
    const token = tokenFor("looping");
    for (let i = 0; i < 3; i++) assert.equal((await send(token)).status, 200);
    const over = await send(token);
    assert.equal(over.status, 429);
    assert.equal((await over.json()).error.code, "too_many_requests");
    assert.equal((await send(tokenFor("other"))).status, 200, "the cap is per run token");
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});
