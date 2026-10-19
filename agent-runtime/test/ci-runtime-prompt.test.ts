import assert from "node:assert/strict";
import { test } from "node:test";

import type { RepairTarget } from "../src/ci-repair.js";
import { CI_REPAIR_DIAGNOSTIC_LIMIT, ciRepairPrompt } from "../src/ci-runtime.js";

const target = (diagnostic: string): RepairTarget => ({
  repository: "acme/widgets", pr: 7, head: "a".repeat(40), tested: "b".repeat(40), branch: "fix", failed_jobs: ["unit"],
  workflow: "tests", run: 11, attempt: 1, diagnostic,
});

test("the CI repair prompt redacts credentials in the diagnostic", () => {
  const prompt = ciRepairPrompt(target("step failed\nAuthorization: Bearer ghs_live_token_value\nGITHUB_TOKEN=ghp_secretvalue123\n"));
  assert.match(prompt, /acme\/widgets PR #7/);
  assert.match(prompt, /step failed/);
  assert.doesNotMatch(prompt, /ghs_live_token_value/);
  assert.doesNotMatch(prompt, /ghp_secretvalue123/);
  assert.match(prompt, /\[REDACTED\]/);
});

test("the CI repair prompt caps the diagnostic length", () => {
  const empty = ciRepairPrompt(target(""));
  const prompt = ciRepairPrompt(target("x".repeat(CI_REPAIR_DIAGNOSTIC_LIMIT * 3)));
  assert.equal(CI_REPAIR_DIAGNOSTIC_LIMIT, 12_000);
  assert.equal(prompt.length, empty.length + CI_REPAIR_DIAGNOSTIC_LIMIT);
});

test("a signed gateway token straddling the length cap is redacted before the cut, so its claims never leak", () => {
  const payload = Buffer.from(JSON.stringify({ runId: "run-secret-claims", expiresAt: 1 })).toString("base64url");
  const token = `${payload}.${"s".repeat(43)}`;
  const filler = "x".repeat(CI_REPAIR_DIAGNOSTIC_LIMIT - payload.length - 10);
  const prompt = ciRepairPrompt(target(`${filler} ${token}`));
  assert.doesNotMatch(prompt, new RegExp(payload.slice(0, 20)));
});
