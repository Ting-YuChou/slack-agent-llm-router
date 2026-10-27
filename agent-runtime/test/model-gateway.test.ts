import assert from "node:assert/strict";
import { once } from "node:events";
import type { AddressInfo } from "node:net";
import { test, type TestContext } from "node:test";

import { SpanStatusCode } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";

import { JEV_CLASSIFIER_MODEL, resolveAgentModel } from "../src/agent-model.js";
import type { BudgetAdmissionLedger } from "../src/budget-rpc.js";
import { issueClassifierGatewayToken, issueGatewayToken, type ClassifierGatewayClaims, type GatewayClaims } from "../src/gateway-token.js";
import { createModelGateway, validateAgentModelRequest } from "../src/model-gateway.js";
import { CREDENTIAL_PLACEHOLDER } from "../src/run-credentials.js";
import { AgentTracing } from "../src/tracing.js";

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

test("the container's credential placeholder never reaches a provider", async () => {
  let calls = 0;
  const gateway = createModelGateway({
    signingSecret: "signing-secret",
    providerApiKeys: { openai: "real-openai" },
    fetchFn: async () => { calls++; return new Response("{}", { status: 200, headers: { "content-type": "application/json" } }); },
  });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  const port = (gateway.address() as AddressInfo).port;
  try {
    const response = await fetch(`http://127.0.0.1:${port}/openai/v1/responses`, {
      method: "POST",
      headers: { authorization: `Bearer ${CREDENTIAL_PLACEHOLDER}`, "content-type": "application/json" },
      body: JSON.stringify({ model: "gpt-5.6-luna", input: [] }),
    });
    assert.equal(response.status, 401);
    assert.equal(calls, 0);
  } finally {
    gateway.close();
    await once(gateway, "close");
  }
});

type GatewayOptions = Partial<Parameters<typeof createModelGateway>[0]>;
const luna = resolveAgentModel("openai/gpt-5.6-luna");
const JEV_REF = `${JEV_CLASSIFIER_MODEL.provider}/${JEV_CLASSIFIER_MODEL.id}`;
const JEV_QUESTION = { model: JEV_CLASSIFIER_MODEL.id, state: { text: "x" }, questions: { kind: { type: "choice", instructions: "Classify.", criteria: { bug: "b", feature: "f" } } } };

async function gatewayAt(t: TestContext, options: GatewayOptions): Promise<string> {
  const gateway = createModelGateway({ signingSecret: "signing-secret", providerApiKeys: { openai: "real-openai", openrouter: "real-openrouter" },
    fetchFn: async () => new Response("{}", { status: 200 }), ...options });
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  t.after(async () => { gateway.close(); await once(gateway, "close"); });
  return `http://127.0.0.1:${(gateway.address() as AddressInfo).port}`;
}
function agentRequest(base: string, claims: Partial<GatewayClaims> = {}): Promise<Response> {
  const token = issueGatewayToken({ runId: "r", provider: luna.provider, model: luna.id, api: luna.api, reasoningEffort: "max", expiresAt: Date.now() + 60_000, ...claims }, "signing-secret");
  return fetch(`${base}${luna.gatewayPath}`, { method: "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, body: JSON.stringify({ model: luna.id, reasoning: { effort: "max" }, input: "x" }) });
}
function classifierRequest(base: string, claims: Partial<ClassifierGatewayClaims> = {}): Promise<Response> {
  const token = issueClassifierGatewayToken({ kind: "classifier", runId: "c", provider: "openrouter", model: JEV_CLASSIFIER_MODEL.id, api: JEV_CLASSIFIER_MODEL.api, maxCalls: 5, expiresAt: Date.now() + 60_000, ...claims }, "signing-secret");
  return fetch(`${base}${JEV_CLASSIFIER_MODEL.gatewayPath}`, { method: "POST", headers: { authorization: `Bearer ${token}`, "content-type": "application/json" }, body: JSON.stringify(JEV_QUESTION) });
}
/** A budget store that refuses every reservation with `error`; `active` answers the classifier's revocation check. */
function refusingLedger(error?: Error, active = true): BudgetAdmissionLedger {
  return { reserve: async () => { if (error) throw error; }, settle: async () => {}, firstModel: async () => undefined, active: async () => active };
}
/** Sends `count` requests that hold their provider stream open until `release`. */
async function holdStreams(t: TestContext, extra: GatewayOptions, count: number) {
  let release!: () => void;
  const held = new Promise<void>((resolve) => { release = resolve; });
  let started = 0;
  let allStarted!: () => void;
  const streaming = new Promise<void>((resolve) => { allStarted = resolve; });
  const base = await gatewayAt(t, { ...extra, fetchFn: async () => { if (++started === count) allStarted(); await held; return new Response("{}", { status: 200 }); } });
  const open = Array.from({ length: count }, () => agentRequest(base));
  await streaming;
  return { base, finish: async () => { release(); await Promise.all((await Promise.all(open)).map((response) => response.text())); } };
}
async function admitted(response: Response): Promise<void> {
  assert.equal(response.status, 200);
  await response.text();
}

/** Every way the gateway answers 429, the request that provokes it, and the one cause it must name. */
const REFUSALS: Array<{ name: string; code: string; cause: string; refuse(t: TestContext, extra: GatewayOptions): Promise<Response> }> = [
  { name: "an agent request over an exhausted budget", code: "workflow_budget_unavailable_or_exhausted", cause: "exhausted",
    refuse: async (t, extra) => agentRequest(await gatewayAt(t, { ...extra, budgetLedger: refusingLedger(new Error("workflow_budget_exhausted")) }), { workflowRunId: "wf" }) },
  { name: "an agent request over an exhausted discovery stage", code: "workflow_budget_unavailable_or_exhausted", cause: "exhausted",
    refuse: async (t, extra) => agentRequest(await gatewayAt(t, { ...extra, budgetLedger: refusingLedger(new Error("discovery_budget_exhausted")) }), { workflowRunId: "wf" }) },
  { name: "an agent token bound to no budget", code: "workflow_budget_unavailable_or_exhausted", cause: "exhausted",
    refuse: async (t, extra) => agentRequest(await gatewayAt(t, { ...extra, budgetLedger: refusingLedger() })) },
  { name: "an agent request the budget store cannot answer", code: "workflow_budget_unavailable_or_exhausted", cause: "unavailable",
    refuse: async (t, extra) => agentRequest(await gatewayAt(t, { ...extra, budgetLedger: refusingLedger(new TypeError("fetch failed")) }), { workflowRunId: "wf" }) },
  { name: "an agent request past the token's request cap", code: "too_many_requests", cause: "request_cap",
    refuse: async (t, extra) => { const base = await gatewayAt(t, { ...extra, maxRequestsPerToken: 1 }); await admitted(await agentRequest(base)); return agentRequest(base); } },
  { name: "a fifth concurrent stream", code: "too_many_streams", cause: "too_many_streams",
    refuse: async (t, extra) => { const streams = await holdStreams(t, extra, 4); const refused = await agentRequest(streams.base); await streams.finish(); return refused; } },
  { name: "a classifier token whose run has ended", code: "token_revoked", cause: "exhausted",
    refuse: async (t, extra) => classifierRequest(await gatewayAt(t, { ...extra, budgetLedger: refusingLedger(undefined, false) })) },
  { name: "a classifier token past its call cap", code: "classifier_call_budget_exhausted", cause: "request_cap",
    refuse: async (t, extra) => { const base = await gatewayAt(t, extra); await admitted(await classifierRequest(base, { maxCalls: 1 })); return classifierRequest(base, { maxCalls: 1 }); } },
  { name: "a workflow classifier request over an exhausted budget", code: "workflow_budget_unavailable_or_exhausted", cause: "exhausted",
    refuse: async (t, extra) => classifierRequest(await gatewayAt(t, { ...extra, prices: { [JEV_REF]: { input: 1, output: 1 } }, budgetLedger: refusingLedger(new Error("workflow_budget_exhausted")) }), { workflowRunId: "wf" }) },
  { name: "a workflow classifier request the gateway cannot price", code: "workflow_budget_unavailable_or_exhausted", cause: "unavailable",
    refuse: async (t, extra) => classifierRequest(await gatewayAt(t, { ...extra, prices: {}, budgetLedger: refusingLedger() }), { workflowRunId: "wf" }) },
];

test("every gateway 429 names one cause beside its unchanged code", { timeout: 10_000 }, async (t) => {
  for (const refusal of REFUSALS) {
    const response = await refusal.refuse(t, {});
    assert.equal(response.status, 429, refusal.name);
    const { error } = await response.json() as { error: { code: string; cause?: string } };
    assert.deepEqual({ code: error.code, cause: error.cause }, { code: refusal.code, cause: refusal.cause }, refusal.name);
  }
});

test("every gateway 429 leaves a failed request span carrying its cause_code", { timeout: 10_000 }, async (t) => {
  for (const refusal of REFUSALS) {
    const exporter = new InMemorySpanExporter();
    const tracing = new AgentTracing([new SimpleSpanProcessor(exporter)]);
    const response = await refusal.refuse(t, { tracing });
    assert.equal(response.status, 429, refusal.name);
    await response.text();
    const refused = exporter.getFinishedSpans().filter((span) => span.attributes["http.response.status_code"] === 429);
    assert.equal(refused.length, 1, refusal.name);
    assert.equal(refused[0].name, "gen_ai.request", refusal.name);
    assert.equal(refused[0].attributes["agent.error.cause_code"], refusal.cause, refusal.name);
    assert.equal(refused[0].attributes["agent.error.code"], refusal.code, refusal.name);
    assert.equal(refused[0].status.code, SpanStatusCode.ERROR, refusal.name);
    await tracing.close();
  }
});

const INCOMPLETE = 'data: {"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":5,"output_tokens":9}}}\n\n';
const COMPLETED = 'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":9}}}\n\n';
/** A budget store that admits everything and records what the gateway reports as truncated. */
function truncationLedger(reports: Array<[string, string | undefined]>): BudgetAdmissionLedger {
  return { reserve: async () => {}, settle: async () => {}, firstModel: async () => undefined, truncated: async (execution, role) => { reports.push([execution, role]); } };
}

test("a response cut off by its output cap is reported to the host with the token's role", async (t) => {
  const reports: Array<[string, string | undefined]> = [];
  const exporter = new InMemorySpanExporter();
  const tracing = new AgentTracing([new SimpleSpanProcessor(exporter)]);
  t.after(() => tracing.close());
  let body = INCOMPLETE;
  const base = await gatewayAt(t, { tracing, prices: { "openai/gpt-5.6-luna": { input: 1, output: 1 } }, budgetLedger: truncationLedger(reports), fetchFn: async () => new Response(body, { status: 200 }) });
  await admitted(await agentRequest(base, { runId: "exec-w", workflowRunId: "wf", role: "writer", outputTokenLimit: 32768 }));
  await admitted(await agentRequest(base, { runId: "exec-legacy", workflowRunId: "wf" }));
  body = COMPLETED;
  await admitted(await agentRequest(base, { runId: "exec-g", workflowRunId: "wf", role: "grader", outputTokenLimit: 16384 }));
  // The report is sent once the response has ended, so wait for the gateway's request spans to close.
  for (let i = 0; i < 50 && exporter.getFinishedSpans().length < 3; i++) await new Promise((resolve) => setTimeout(resolve, 10));
  assert.deepEqual(reports, [["exec-w", "writer"], ["exec-legacy", undefined]], "a complete response is not reported; a role-less token reports no role");
  assert.equal(exporter.getFinishedSpans().filter((span) => span.attributes["gen_ai.response.truncated"] === true).length, 2);
});

test("a host that cannot record a truncation does not fail the request", async (t) => {
  const ledger: BudgetAdmissionLedger = { ...truncationLedger([]), truncated: async () => { throw new Error("workflow_budget_unavailable"); } };
  const base = await gatewayAt(t, { prices: { "openai/gpt-5.6-luna": { input: 1, output: 1 } }, budgetLedger: ledger, fetchFn: async () => new Response(INCOMPLETE, { status: 200 }) });
  const response = await agentRequest(base, { workflowRunId: "wf", role: "writer", outputTokenLimit: 32768 });
  assert.equal(response.status, 200);
  assert.equal(await response.text(), INCOMPLETE);
});

test("the gateway applies a token's output cap and never sends one above the ceiling", async (t) => {
  const bodies: Array<Record<string, unknown>> = [];
  const base = await gatewayAt(t, { prices: { "openai/gpt-5.6-luna": { input: 1, output: 1 } }, budgetLedger: truncationLedger([]),
    fetchFn: async (_input, init) => { bodies.push(JSON.parse(String(init?.body))); return new Response(COMPLETED, { status: 200 }); } });
  await admitted(await agentRequest(base, { workflowRunId: "wf", role: "writer", outputTokenLimit: 12000 }));
  await admitted(await agentRequest(base, { workflowRunId: "wf", role: "writer", outputTokenLimit: 100_000 }));
  assert.deepEqual(bodies.map((body) => body.max_output_tokens), [12000, 32768]);
});
