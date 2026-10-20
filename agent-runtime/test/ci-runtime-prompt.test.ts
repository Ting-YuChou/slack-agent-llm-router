import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { ControlStore, INBOX_MAX_ATTEMPTS } from "../src/ci-control.js";

import type { RepairTarget } from "../src/ci-repair.js";
import { CI_REPAIR_DIAGNOSTIC_LIMIT, ciRepairPrompt, recordInboxFailure } from "../src/ci-runtime.js";

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

test("a transient inbox failure is retried with backoff until it goes dead; a terminal reason finishes the row", () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "ci-inbox-failure-"));
  try {
    const store = new ControlStore(path.join(root, "control.sqlite"));
    store.enqueue("transient", "repo:run:1:sha", {});
    store.enqueue("terminal", "repo:run:2:sha", {});
    let now = 0;
    for (let attempt = 1; attempt <= INBOX_MAX_ATTEMPTS; attempt++) {
      assert.ok(store.pending(now).some(row => row.delivery === "transient"), `attempt ${attempt} is due`);
      recordInboxFailure(store, "transient", new Error("github_unavailable"), now);
      assert.deepEqual(store.get("inbox_error", "transient"), { reason: "github_unavailable", attempts: attempt, status: attempt === INBOX_MAX_ATTEMPTS ? "dead" : "pending", at: new Date(now).toISOString() });
      now += 300_000;
    }
    assert.equal(store.pending(Number.MAX_SAFE_INTEGER).some(row => row.delivery === "transient"), false);
    recordInboxFailure(store, "terminal", new Error("pr_not_found"), 0);
    assert.deepEqual(store.get("inbox_error", "terminal"), { reason: "pr_not_found", attempts: 0, status: "done", at: new Date(0).toISOString() });
    assert.deepEqual(store.pending(Number.MAX_SAFE_INTEGER), []);
    store.close();
  } finally { rmSync(root, { recursive: true, force: true }); }
});
