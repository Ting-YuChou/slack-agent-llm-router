import assert from "node:assert/strict";
import test from "node:test";

import { HarnessError } from "../src/error-taxonomy.js";
import { QualityGate, type Grade, type GraderSampling } from "../src/quality-gate.js";
import { validateProfile } from "../src/verification.js";

const base = { id: "p", repository: "o/r", image: `sha256:${"0".repeat(64)}`, rubric: ["ok"], allowedPaths: ["src"], checks: [{ name: "t", argv: ["true"], cwd: ".", timeoutMs: 1000 }] };

test("a profile grades once and unanimously unless it asks for more samples", () => {
  assert.deepEqual(validateProfile(base).grader, { samples: 1, aggregate: "unanimous" });
  assert.deepEqual(validateProfile({ ...base, grader: {} }).grader, { samples: 1, aggregate: "unanimous" });
  assert.deepEqual(validateProfile({ ...base, grader: { samples: 3 } }).grader, { samples: 3, aggregate: "unanimous" });
  assert.deepEqual(validateProfile({ ...base, grader: { samples: 3, aggregate: "majority" } }).grader, { samples: 3, aggregate: "majority" });
  assert.deepEqual(validateProfile({ ...base, grader: { samples: 1, aggregate: "majority" } }).grader, { samples: 1, aggregate: "majority" });
});

test("a profile with any other grader sampling is refused", () => {
  for (const grader of [null, "3", [], { samples: 2 }, { samples: 0 }, { samples: 5 }, { samples: "3" }, { samples: 3.0001 },
    { aggregate: "any" }, { aggregate: "" }, { samples: 3, aggregate: "Majority" }, { samples: 3, extra: true }]) {
    assert.throws(() => validateProfile({ ...base, grader }), /Invalid grader sampling/, JSON.stringify(grader));
  }
});

type Step = "passed" | "needs_changes" | "unknown" | "infra" | "invalid";
const criteria = (passed: boolean) => [{ id: "correct", passed, ...(passed ? { evidence: "src/a.ts:1 ok" } : {}) }];
/** A gate whose grader plays one step per execution; it counts executions and keeps every recorded payload. */
function sampledGate(sampling: GraderSampling | undefined, steps: Step[]) {
  let executions = 0;
  const recorded: Array<{ kind: string; payload: any }> = [];
  const gate = new QualityGate({
    profile: { id: "p", rubric: ["correct"], maxRepairs: 2, ...(sampling ? { grader: sampling } : {}) },
    snapshot: async () => ({ hash: "hash", path: "/snapshot" }),
    verify: async () => ({ status: "passed", checks: [] }),
    grade: async () => {
      const step = steps[executions++];
      if (step === undefined) throw new Error(`unexpected grader execution ${executions}`);
      if (step === "infra") throw new HarnessError("grader_timeout", "slow");
      if (step === "invalid") return { status: "passed", candidate_hash: "other", criteria: [], findings: [] };
      return { status: step, candidate_hash: "hash", criteria: criteria(step === "passed"),
        findings: step === "needs_changes" ? [{ path: "src/a.ts", line: executions, severity: "error", reason: `sample ${executions}` }] : [] };
    },
    record: async (kind, payload) => { recorded.push({ kind, payload }); },
  });
  const input = { run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: new AbortController().signal };
  return { evaluate: () => gate.evaluate(input), executions: () => executions, grade: () => recorded.find(r => r.kind === "grade")?.payload as Grade & Record<string, any> };
}
const verdicts = (grade: Grade | undefined) => grade?.sample_verdicts?.map(sample => sample.status);

test("unanimous sampling passes only when every sample passes", async () => {
  const gate = sampledGate({ samples: 3, aggregate: "unanimous" }, ["passed", "passed", "passed"]);
  const result = await gate.evaluate();
  assert.equal(result.action, "pass");
  assert.equal(result.grade?.status, "passed");
  assert.equal(result.grade?.cause, undefined);
  assert.equal(gate.executions(), 3);
  assert.equal(result.grade?.grader_attempts, 3);
});

test("unanimous sampling stops at the first needs_changes and repairs with that sample's findings", async () => {
  const gate = sampledGate({ samples: 3, aggregate: "unanimous" }, ["passed", "needs_changes", "passed"]);
  const result = await gate.evaluate();
  assert.equal(gate.executions(), 2);
  assert.equal(result.action, "repair");
  assert.equal(result.grade?.status, "needs_changes");
  assert.equal(result.grade?.cause, undefined);
  assert.deepEqual(result.grade?.findings, [{ path: "src/a.ts", line: 2, severity: "error", reason: "sample 2" }]);
  assert.deepEqual(verdicts(result.grade), ["passed", "needs_changes"]);
  assert.match(result.repair_prompt ?? "", /src\/a\.ts:2 \[error\] sample 2/);

  const first = sampledGate({ samples: 3, aggregate: "unanimous" }, ["needs_changes"]);
  assert.equal((await first.evaluate()).grade?.status, "needs_changes");
  assert.equal(first.executions(), 1);
});

test("unanimous sampling keeps sampling past an unknown sample while needs_changes is still possible", async () => {
  const split = sampledGate({ samples: 3, aggregate: "unanimous" }, ["unknown", "passed", "needs_changes"]);
  const result = await split.evaluate();
  assert.equal(split.executions(), 3);
  assert.equal(result.grade?.status, "needs_changes");
  assert.equal(result.action, "repair");

  const undecided = sampledGate({ samples: 3, aggregate: "unanimous" }, ["passed", "unknown", "passed"]);
  const unknown = await undecided.evaluate();
  assert.equal(undecided.executions(), 3);
  assert.equal(unknown.grade?.status, "unknown");
  assert.equal(unknown.grade?.cause?.code, "insufficient_evidence");
  assert.deepEqual(unknown.failure, { code: "verification_failed", cause_code: "insufficient_evidence" });
  assert.equal(unknown.action, "fail");
});

test("majority sampling decides on a strict majority of the valid samples and stops once it is settled", async () => {
  const settled = sampledGate({ samples: 3, aggregate: "majority" }, ["passed", "passed", "needs_changes"]);
  assert.equal((await settled.evaluate()).grade?.status, "passed");
  assert.equal(settled.executions(), 2);

  const rejected = sampledGate({ samples: 3, aggregate: "majority" }, ["needs_changes", "needs_changes", "passed"]);
  assert.equal((await rejected.evaluate()).grade?.status, "needs_changes");
  assert.equal(rejected.executions(), 2);

  const split = sampledGate({ samples: 3, aggregate: "majority" }, ["passed", "needs_changes", "needs_changes"]);
  const result = await split.evaluate();
  assert.equal(split.executions(), 3);
  assert.equal(result.grade?.status, "needs_changes");
  assert.deepEqual(result.grade?.findings, [{ path: "src/a.ts", line: 2, severity: "error", reason: "sample 2" }]);

  const late = sampledGate({ samples: 3, aggregate: "majority" }, ["needs_changes", "passed", "passed"]);
  const passed = await late.evaluate();
  assert.equal(late.executions(), 3);
  assert.equal(passed.grade?.status, "passed");
  assert.equal(passed.grade?.criteria[0]?.passed, true);
  assert.equal(passed.grade?.findings.length, 0);
});

test("majority sampling counts only valid samples, so an unknown sample cannot outvote them", async () => {
  // Two valid samples agree: a strict majority of the valid ones.
  const gate = sampledGate({ samples: 3, aggregate: "majority" }, ["unknown", "passed", "passed"]);
  const result = await gate.evaluate();
  assert.equal(gate.executions(), 3);
  assert.equal(result.grade?.status, "passed");
  assert.deepEqual(verdicts(result.grade), ["unknown", "passed", "passed"]);
});

test("majority sampling is unknown with a cause only when the valid samples tie", async () => {
  const tie = sampledGate({ samples: 3, aggregate: "majority" }, ["passed", "needs_changes", "unknown"]);
  const result = await tie.evaluate();
  assert.equal(tie.executions(), 3);
  assert.equal(result.grade?.status, "unknown");
  assert.equal(result.grade?.cause?.code, "insufficient_evidence");
  assert.equal(result.action, "fail");

  const none = sampledGate({ samples: 3, aggregate: "majority" }, ["invalid", "invalid", "unknown", "infra", "infra"]);
  const invalid = await none.evaluate();
  assert.equal(none.executions(), 5);
  assert.equal(invalid.grade?.status, "unknown");
  // The first sample that gave no verdict names the cause.
  assert.equal(invalid.grade?.cause?.code, "grader_invalid_submission");
  assert.deepEqual(invalid.failure, { code: "verification_failed", cause_code: "grader_invalid_submission" });
  assert.deepEqual(invalid.grade?.criteria, [{ id: "correct", passed: false }]);
});

test("each sample keeps its own grader_infra_error retry", async () => {
  const gate = sampledGate({ samples: 3, aggregate: "unanimous" }, ["passed", "infra", "passed", "infra", "infra"]);
  const result = await gate.evaluate();
  assert.equal(gate.executions(), 5);
  assert.equal(result.grade?.grader_attempts, 5);
  assert.deepEqual(result.grade?.sample_verdicts, [
    { status: "passed", grader_attempts: 1 },
    { status: "passed", grader_attempts: 2 },
    { status: "unknown", grader_attempts: 2, cause_code: "grader_timeout" },
  ]);
  assert.equal(result.grade?.status, "unknown");
  assert.equal(result.grade?.cause?.code, "grader_timeout");
});

test("each grade record carries the sampling and every sample's verdict", async () => {
  // The second sample's first submission is rejected, so its retry supplies the verdict.
  const gate = sampledGate({ samples: 3, aggregate: "majority" }, ["passed", "invalid", "needs_changes", "passed"]);
  const result = await gate.evaluate();
  const recorded = gate.grade();
  assert.equal(recorded.status, "passed");
  assert.equal(recorded.samples, 3);
  assert.equal(recorded.aggregate, "majority");
  assert.deepEqual(recorded.sample_verdicts, [
    { status: "passed", grader_attempts: 1 },
    { status: "needs_changes", grader_attempts: 2 },
    { status: "passed", grader_attempts: 1 },
  ]);
  assert.equal(recorded.grader_attempts, 4);
  assert.deepEqual(result.grade?.sample_verdicts, recorded.sample_verdicts);
});

test("the default single sample grades once and records one sample", async () => {
  for (const sampling of [undefined, { samples: 1, aggregate: "unanimous" } as const, { samples: 1, aggregate: "majority" } as const]) {
    const gate = sampledGate(sampling, ["needs_changes", "passed"]);
    const result = await gate.evaluate();
    assert.equal(gate.executions(), 1);
    assert.equal(result.grade?.status, "needs_changes");
    assert.equal(result.grade?.grader_attempts, 1);
    assert.equal(gate.grade().samples, 1);
    assert.deepEqual(gate.grade().sample_verdicts, [{ status: "needs_changes", grader_attempts: 1 }]);
  }
});

test("an abort or exhausted budget during a later sample stops grading instead of voting", async () => {
  const controller = new AbortController();
  let executions = 0;
  const make = (fail: () => never) => new QualityGate({
    profile: { id: "p", rubric: ["correct"], maxRepairs: 2, grader: { samples: 3, aggregate: "unanimous" } },
    snapshot: async () => ({ hash: "hash", path: "/snapshot" }),
    verify: async () => ({ status: "passed", checks: [] }),
    grade: async () => (++executions === 1 ? { status: "passed", candidate_hash: "hash", criteria: criteria(true), findings: [] } : fail()),
  });
  const input = { run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: controller.signal };
  await assert.rejects(make(() => { throw new Error("workflow_budget_exhausted"); }).evaluate(input), /workflow_budget_exhausted/);
  assert.equal(executions, 2);
  executions = 0;
  await assert.rejects(make(() => { controller.abort(); throw new Error("grader_aborted"); }).evaluate(input));
  assert.equal(executions, 2);
});
