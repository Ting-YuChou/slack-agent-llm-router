import assert from "node:assert/strict";
import test from "node:test";

import { ScopeViolationError } from "../src/candidate-scope.js";
import { HarnessError } from "../src/error-taxonomy.js";
import { QualityGate, type GraderSampling } from "../src/quality-gate.js";

type GradeStep = { ms: number; status?: "passed" | "needs_changes"; infra?: true };
/** A gate on a scripted clock: every stage advances the clock by the time it is said to take. */
function timedGate(script: { snapshotMs: number; snapshotError?: Error; verifyMs?: number; verification?: "passed" | "failed"; grades?: GradeStep[]; sampling?: GraderSampling }) {
  let clock = 1000; let graded = 0;
  const gate = new QualityGate({
    profile: { id: "p", rubric: ["correct"], maxRepairs: 2, ...(script.sampling ? { grader: script.sampling } : {}) },
    now: () => clock,
    snapshot: async () => { clock += script.snapshotMs; if (script.snapshotError) throw script.snapshotError; return { hash: "hash", path: "/snapshot" }; },
    verify: async () => { clock += script.verifyMs ?? 0; return { status: script.verification ?? "passed", checks: [] }; },
    grade: async () => {
      const step = script.grades?.[graded++];
      if (!step) throw new Error(`unexpected grader execution ${graded}`);
      clock += step.ms;
      if (step.infra) throw new HarnessError("grader_timeout", "slow");
      const passed = (step.status ?? "passed") === "passed";
      return { status: step.status ?? "passed", candidate_hash: "hash", criteria: [{ id: "correct", passed, ...(passed ? { evidence: "src/a.ts:1 ok" } : {}) }],
        findings: passed ? [] : [{ path: "src/a.ts", line: 1, severity: "error", reason: "wrong" }] };
    },
    // Recording takes time too; it belongs to no stage.
    record: async () => { clock += 7; },
  });
  return gate.evaluate({ run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: new AbortController().signal });
}

test("a passing gate reports snapshot, verify and per-sample grade time", async () => {
  const result = await timedGate({ snapshotMs: 120, verifyMs: 3400, sampling: { samples: 3, aggregate: "unanimous" }, grades: [{ ms: 1000 }, { ms: 2000 }, { ms: 1500 }] });
  assert.equal(result.action, "pass");
  assert.deepEqual(result.timing, { snapshot_ms: 120, verify_ms: 3400, grade_ms: 4500, grade_samples_ms: [1000, 2000, 1500] });
});

test("a grader replaced after an infrastructure error counts toward the same sample", async () => {
  const result = await timedGate({ snapshotMs: 10, verifyMs: 20, grades: [{ ms: 600, infra: true }, { ms: 400 }] });
  assert.equal(result.action, "pass");
  assert.deepEqual(result.timing, { snapshot_ms: 10, verify_ms: 20, grade_ms: 1000, grade_samples_ms: [1000] });
});

test("a failed verification has no grade time", async () => {
  const result = await timedGate({ snapshotMs: 50, verifyMs: 900, verification: "failed" });
  assert.equal(result.action, "repair");
  assert.deepEqual(result.timing, { snapshot_ms: 50, verify_ms: 900 });
});

test("a scope violation reports snapshot time only", async () => {
  const result = await timedGate({ snapshotMs: 80, snapshotError: new ScopeViolationError(["docs/notes.md"]) });
  assert.equal(result.failure?.code, "scope_violation");
  assert.deepEqual(result.timing, { snapshot_ms: 80 });
});

test("stage times are whole milliseconds on the default clock", async () => {
  const gate = new QualityGate({
    profile: { id: "p", rubric: ["correct"], maxRepairs: 2 },
    snapshot: async () => ({ hash: "hash", path: "/snapshot" }),
    verify: async () => ({ status: "failed", checks: [] }),
    grade: async () => { throw new Error("not reached"); },
  });
  const { timing } = await gate.evaluate({ run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: new AbortController().signal });
  assert.ok(timing && Number.isInteger(timing.snapshot_ms) && timing.snapshot_ms >= 0);
  assert.ok(Number.isInteger(timing.verify_ms) && timing.verify_ms! >= 0);
});
