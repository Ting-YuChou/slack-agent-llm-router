import assert from "node:assert/strict";
import { test } from "node:test";

import {
  issueClassifierGatewayToken,
  issueGatewayToken,
  verifyClassifierGatewayToken,
  verifyGatewayToken,
} from "../src/gateway-token.js";

test("gateway token is short-lived and bound to run, provider, model, API, and effort", () => {
  const token = issueGatewayToken({
    runId: "run-1",
    provider: "anthropic",
    model: "claude-sonnet-4-6",
    api: "anthropic-messages",
    reasoningEffort: "max",
    expiresAt: 2_000,
  }, "secret");
  assert.deepEqual(verifyGatewayToken(token, "secret", 1_000), {
    runId: "run-1",
    provider: "anthropic",
    model: "claude-sonnet-4-6",
    api: "anthropic-messages",
    reasoningEffort: "max",
    expiresAt: 2_000,
  });
  assert.equal(verifyGatewayToken(token, "secret", 2_001), null);
  assert.equal(verifyGatewayToken(`${token}x`, "secret", 1_000), null);
});

test("gateway refuses token claims outside the allowlisted model registry", () => {
  const token = issueGatewayToken({
    runId: "run-1",
    provider: "openai",
    model: "other",
    api: "openai-responses",
    reasoningEffort: "max",
    expiresAt: 2_000,
  }, "secret");
  assert.equal(verifyGatewayToken(token, "secret", 1_000), null);
});

test("gateway accepts Sol high and rejects low effort for an unsupported provider", () => {
  const sol = issueGatewayToken({ runId: "r", provider: "openai", model: "gpt-5.6-sol", api: "openai-responses", reasoningEffort: "high", expiresAt: 2000 } as any, "secret");
  const anthropic = issueGatewayToken({ runId: "r", provider: "anthropic", model: "claude-sonnet-4-6", api: "anthropic-messages", reasoningEffort: "low", expiresAt: 2000 } as any, "secret");
  assert.equal(verifyGatewayToken(sol, "secret", 1000)?.reasoningEffort, "high");
  assert.equal(verifyGatewayToken(anthropic, "secret", 1000), null);
});

test("classifier token is short-lived and bound to the allowlisted Jev operation", () => {
  const token = issueClassifierGatewayToken({
    kind: "classifier",
    runId: "run-classifier",
    provider: "openrouter",
    model: "typesafe/jev-1.13",
    api: "typesafe-system-one",
    maxCalls: 6,
    expiresAt: 2_000,
  }, "secret");
  assert.deepEqual(verifyClassifierGatewayToken(token, "secret", 1_000), {
    kind: "classifier",
    runId: "run-classifier",
    provider: "openrouter",
    model: "typesafe/jev-1.13",
    api: "typesafe-system-one",
    maxCalls: 6,
    expiresAt: 2_000,
  });
  assert.equal(verifyClassifierGatewayToken(token, "secret", 2_001), null);
  assert.equal(verifyClassifierGatewayToken(`${token}x`, "secret", 1_000), null);
});

test("classifier token rejects arbitrary classifier models and invalid call budgets", () => {
  for (const claims of [
    { kind: "classifier", runId: "r", provider: "openrouter", model: "~typesafe/jev-latest", api: "typesafe-system-one", maxCalls: 6, expiresAt: 2_000 },
    { kind: "classifier", runId: "r", provider: "openrouter", model: "typesafe/jev-1.13", api: "typesafe-system-one", maxCalls: 0, expiresAt: 2_000 },
  ]) {
    const token = issueClassifierGatewayToken(claims as any, "secret");
    assert.equal(verifyClassifierGatewayToken(token, "secret", 1_000), null);
  }
});

test("signed optional trace context survives verification and rejects invalid context", () => {
  const claims = {runId:"trace-run",provider:"openai",model:"gpt-5.6-luna",api:"openai-responses",reasoningEffort:"max",expiresAt:Date.now()+60000,
    traceparent:"00-11111111111111111111111111111111-2222222222222222-01"} as any;
  const token=issueGatewayToken(claims,"secret");
  assert.equal((verifyGatewayToken(token,"secret") as any)?.traceparent,claims.traceparent);
  assert.equal(verifyGatewayToken(issueGatewayToken({...claims,traceparent:"bad"},"secret"),"secret"),null);
});
