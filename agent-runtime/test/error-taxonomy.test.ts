import assert from "node:assert/strict";
import { test } from "node:test";

import { classifyGateError, HarnessError } from "../src/error-taxonomy.js";
import { DiffLimitError, ProtectedFileError } from "../src/worktree-manager.js";

test("gate errors keep their stable public codes", () => {
  assert.equal(classifyGateError(new ProtectedFileError("Protected file modification detected: .env")).code, "protected_file_modified");
  assert.equal(classifyGateError(new DiffLimitError("Changed file limit exceeded (51/50)")).code, "diff_limit_exceeded");
  assert.equal(classifyGateError(new Error("workflow_budget_exhausted")).code, "workflow_budget_exhausted");
  const generic = classifyGateError(new Error("something odd"));
  assert.equal(generic.code, "quality_gate_unavailable");
  assert.equal(generic.cause_code, undefined);
  assert.match(generic.message, /something odd/);
});

test("known harness causes ride along on quality_gate_unavailable", () => {
  const failure = classifyGateError(new HarnessError("docker_unavailable", "Cannot connect to the Docker daemon"));
  assert.equal(failure.code, "quality_gate_unavailable");
  assert.equal(failure.cause_code, "docker_unavailable");
  assert.match(failure.message, /Docker daemon/);
});

test("non-Error throws never leak arbitrary text", () => {
  const failure = classifyGateError({ secret: "x" });
  assert.equal(failure.code, "quality_gate_unavailable");
  assert.doesNotMatch(failure.message, /secret/);
});
