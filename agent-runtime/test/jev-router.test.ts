import assert from "node:assert/strict";
import { test } from "node:test";

import { JevRouter } from "../src/jev-router.js";

function decision(choice: string, probability: number) {
  return new Response(JSON.stringify({
    id: "dec-1",
    answers: { task: { type: "choice", choice, probabilities: { [choice]: probability }, confidence: probability } },
    usage: { cost: 0.00002 },
  }), { status: 200 });
}

test("high confidence complex task selects Sol high with only current task sent to OpenRouter", async () => {
  let sent: Record<string, unknown> = {};
  const router = new JevRouter({ mode: "on", apiKey: "test-key", fetchFn: async (_url, init) => {
    sent = JSON.parse(String(init?.body));
    return decision("complex", 0.94);
  } });
  const result = await router.route("Refactor the cache contract across services");
  assert.equal(result.modelRef, "openai/gpt-5.6-sol");
  assert.equal(result.effort, "high");
  assert.equal(result.label, "complex");
  assert.equal(sent.state, "Refactor the cache contract across services");
  assert.equal(sent.model, "typesafe/jev-1.13");
});

test("high confidence routing table selects the configured model and effort", async () => {
  const cases = [
    { label: "read_only", modelRef: "openai/gpt-5.6-luna", effort: "low" },
    { label: "small_patch", modelRef: "openai/gpt-5.6-luna", effort: "medium" },
    { label: "complex", modelRef: "openai/gpt-5.6-sol", effort: "high" },
    { label: "unclear", modelRef: "openai/gpt-5.6-luna", effort: "max" },
  ];
  for (const expected of cases) {
    const router = new JevRouter({ mode: "on", apiKey: "key", fetchFn: async () => decision(expected.label, 0.95) });
    const result = await router.route("current task");
    assert.equal(result.modelRef, expected.modelRef, expected.label);
    assert.equal(result.effort, expected.effort, expected.label);
  }
});

test("low confidence and unavailable Jev preserve Luna max", async () => {
  const low = new JevRouter({ mode: "on", apiKey: "key", fetchFn: async () => decision("complex", 0.89) });
  const unavailable = new JevRouter({ mode: "on", apiKey: "key", fetchFn: async () => { throw new Error("network failed"); } });
  assert.deepEqual((await low.route("Complex task")).modelRef, "openai/gpt-5.6-luna");
  assert.equal((await low.route("Complex task")).effort, "max");
  assert.equal((await low.route("Complex task")).source, "fallback");
  assert.equal((await unavailable.route("Complex task")).modelRef, "openai/gpt-5.6-luna");
});

test("Jev timeout falls back without retrying", async () => {
  let calls = 0;
  const router = new JevRouter({ mode: "on", apiKey: "key", timeoutMs: 5, fetchFn: async (_url, init) => {
    calls++;
    await new Promise((_resolve, reject) => init?.signal?.addEventListener("abort", () => reject(new DOMException("timed out", "TimeoutError"))));
    return decision("complex", 1);
  } });
  const result = await router.route("Complex task");
  assert.equal(result.effort, "max");
  assert.equal(result.reason, "timeout");
  assert.equal(calls, 1);
});

test("shadow records the recommendation without changing execution", async () => {
  const router = new JevRouter({ mode: "shadow", apiKey: "key", fetchFn: async () => decision("read_only", 0.98) });
  const result = await router.route("Find the entrypoint");
  assert.equal(result.modelRef, "openai/gpt-5.6-luna");
  assert.equal(result.effort, "max");
  assert.equal(result.recommendedEffort, "low");
});

test("missing or oversized routing text never calls OpenRouter", async () => {
  let calls = 0;
  const router = new JevRouter({ mode: "on", apiKey: "key", fetchFn: async () => { calls++; return decision("complex", 1); } });
  assert.equal((await router.route(undefined)).effort, "max");
  assert.equal((await router.route("x".repeat(4001))).effort, "max");
  assert.equal(calls, 0);
});
