import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { ControlStore } from "../src/ci-control.js";
import { failureCause, HarnessError } from "../src/error-taxonomy.js";
import { WorkflowCoordinator } from "../src/structured-workflow.js";

function discover(execute: ConstructorParameters<typeof WorkflowCoordinator>[0]["execute"]) {
  const root = mkdtempSync(path.join(os.tmpdir(), "discovery-cause-"));
  const coordinator = new WorkflowCoordinator({ store: new ControlStore(path.join(root, "control.db")), concurrency: 1, execute });
  const input = { run_id: "run-1", owner: "U", requirement: "fix", baseline: "b".repeat(40), snapshot: { hash: "h", path: root }, fingerprint: "f", initial_active_ms: 0 };
  return coordinator.discover(input, new AbortController().signal).finally(() => rmSync(root, { recursive: true, force: true }));
}

const plan = { tasks: [{ id: "explore", role: "code_explorer", purpose: "find it", expected_output: "location", paths: ["src/"], dependencies: [] }] };

test("a node that fails both attempts surfaces the last attempt's cause", async () => {
  const error = await discover(async request => {
    if (request.node.role === "planner") return plan;
    throw new HarnessError("investigation_timeout", "investigation_timeout");
  }).then(() => undefined, (e: unknown) => e);
  assert.equal(failureCause(error), "investigation_timeout");
});

test("a node whose agent reports unknown twice names that cause", async () => {
  const error = await discover(async request => request.node.role === "planner" ? plan
    : { status: "unknown", findings: [], evidence: [], unresolved: ["no access"], suggested_tasks: [] }).then(() => undefined, (e: unknown) => e);
  assert.equal(failureCause(error), "investigation_reported_unknown");
});

test("failureCause reads HarnessError, known messages and nested causes, and nothing else", () => {
  assert.equal(failureCause(new HarnessError("investigation_execution_failed", "x")), "investigation_execution_failed");
  assert.equal(failureCause(new Error("outer", { cause: new Error("outer", { cause: new HarnessError("grader_timeout", "slow") }) })), "grader_timeout");
  assert.equal(failureCause(new Error("something else")), undefined);
  assert.equal(failureCause("not an error"), undefined);
});
