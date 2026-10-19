import assert from "node:assert/strict";
import { test } from "node:test";

import { HarnessError } from "../src/error-taxonomy.js";
import { piTurn, runScenario, settledRecord, type Scenario } from "./support/scenario.js";

const fix = { "src/a.js": "export const a = 2;\n" };
const profile = { allowedPaths: ["src"], maxRepairs: 1 };
const limitEntry = { type: "entry_appended", entry: { type: "custom", customType: "pi_agent_limit", data: { kind: "tool_calls", limit: 40 } } };

/** Harness acceptance contract. A `todo` names the Phase 1 slice that makes the scenario pass. */
const scenarios: Array<Scenario & { todo?: string }> = [
  { name: "S1 verified change commits", profile,
    attempts: [{ files: fix, records: [...piTurn("stop", "Fixed"), settledRecord] }],
    expect: { status: "completed", committed: true, verificationStatus: "passed" } },
  { name: "S2 tool limit abort never commits",
    attempts: [{ files: fix, records: [limitEntry, ...piTurn("aborted"), settledRecord] }],
    expect: { status: "failed", errorCode: "agent_tool_limit", committed: false } },
  { name: "S3 truncated output never commits",
    attempts: [{ files: fix, records: [...piTurn("length", "partial"), settledRecord] }],
    expect: { status: "failed", errorCode: "agent_output_truncated", committed: false } },
  { name: "S4 provider error never commits",
    attempts: [{ files: fix, records: [...piTurn("error", "", { errorMessage: "upstream 500" }), settledRecord] }],
    expect: { status: "failed", errorCode: "provider_error", committed: false } },
  { name: "S5 out-of-scope change is repaired with the violating files listed", profile,
    attempts: [
      { files: { ...fix, "docs/notes.md": "scratch\n" }, records: [...piTurn("stop"), settledRecord] },
      { files: { "docs/notes.md": null }, records: [...piTurn("stop"), settledRecord] },
    ],
    expect: { status: "completed", committed: true, repairPromptIncludes: ["docs/notes.md"] } },
  { name: "S6 diff limit reports its own code", profile,
    attempts: [{ files: Object.fromEntries(Array.from({ length: 51 }, (_, i) => [`src/f${i}.js`, `${i}\n`])), records: [...piTurn("stop"), settledRecord] }],
    expect: { status: "failed", errorCode: "diff_limit_exceeded", committed: false } },
  { name: "S7 no-profile change is labelled unverified",
    attempts: [{ files: fix, records: [...piTurn("stop", "Fixed"), settledRecord] }],
    expect: { status: "completed", committed: true, verificationStatus: "unverified" } },
  { name: "S8 failed verification is repaired then committed", profile, verify: ["failed", "passed"],
    attempts: [
      { files: { "src/a.js": "broken\n" }, records: [...piTurn("stop"), settledRecord] },
      { files: fix, records: [...piTurn("stop"), settledRecord] },
    ],
    expect: { status: "completed", committed: true, verificationStatus: "passed" } },
  { name: "S9 unknown grade never commits", profile, grade: ["unknown"],
    attempts: [{ files: fix, records: [...piTurn("stop"), settledRecord] }],
    expect: { status: "failed", errorCode: "verification_failed", committed: false } },
];

for (const scenario of scenarios) {
  test(scenario.name, { todo: scenario.todo }, async () => {
    const { run, prompts, commits, dirty } = await runScenario(scenario);
    const { expect } = scenario;
    assert.equal(run.status, expect.status, JSON.stringify(run.error));
    if (expect.errorCode) assert.equal(run.error?.code, expect.errorCode);
    assert.equal(commits, expect.committed ? 1 : 0);
    if (!expect.committed) assert.equal(dirty, "", "failed runs must leave the worktree rolled back");
    if (expect.verificationStatus) assert.equal(run.verification_status, expect.verificationStatus);
    for (const text of expect.repairPromptIncludes ?? []) assert.match(prompts[1] ?? "", new RegExp(text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  });
}

test("harness: full verification results reach the repair prompt", async () => {
  const failed = { status: "failed" as const, checks: [{ name: "candidate:pytest", status: "failed" as const, exit_code: 1, failed: 3 }] };
  const { run, prompts } = await runScenario({ name: "H0 verify result", profile, verify: [failed, "passed"],
    attempts: [
      { files: { "src/a.js": "broken\n" }, records: [...piTurn("stop"), settledRecord] },
      { files: fix, records: [...piTurn("stop"), settledRecord] },
    ],
    expect: { status: "completed", committed: true } });
  assert.equal(run.status, "completed");
  assert.match(prompts[1] ?? "", /candidate:pytest/);
});

test("harness: grader steps are consumed per grader call and may throw", async () => {
  const { run, gradeCalls } = await runScenario({ name: "H0 grader error", profile, grade: [new Error("grader broke"), new Error("grader broke again")],
    attempts: [{ files: fix, records: [...piTurn("stop"), settledRecord] }],
    expect: { status: "failed", committed: false } });
  assert.equal(run.status, "failed");
  assert.equal(gradeCalls, 2);
});

test("harness: quality overrides replace scripted services", async () => {
  let archived = 0;
  const { run } = await runScenario({ name: "H0 quality override", profile, verify: ["failed", "failed"],
    quality: { archive: async () => { archived += 1; } },
    attempts: [{ files: { "src/a.js": "broken\n" }, records: [...piTurn("stop"), settledRecord] }, { files: { "src/a.js": "broken\n" }, records: [...piTurn("stop"), settledRecord] }],
    expect: { status: "failed", committed: false } });
  assert.equal(run.status, "failed");
  assert.ok(archived >= 1, "failed runs archive through the overridden service");
});

const unknownSubmission = (_call: number, hash: string) => ({ status: "unknown", candidate_hash: hash, criteria: [{ id: "meets_request", passed: false }], findings: [] });
const graderScenarios: Array<{ name: string; grade: NonNullable<Scenario["grade"]>; status: string; gradeCalls: number; cause?: string }> = [
  { name: "S11 a grader timeout is retried and the change commits", grade: [new HarnessError("grader_timeout", "slow"), "passed"], status: "completed", gradeCalls: 2 },
  { name: "S12 two grader timeouts fail with grader_timeout", grade: [new HarnessError("grader_timeout", "slow"), new HarnessError("grader_timeout", "slow")], status: "failed", gradeCalls: 2, cause: "grader_timeout" },
  { name: "S13 a grader that submits unknown fails once as insufficient_evidence", grade: [unknownSubmission], status: "failed", gradeCalls: 1, cause: "insufficient_evidence" },
];
for (const scenario of graderScenarios) {
  test(scenario.name, async () => {
    const { run, commits, gradeCalls } = await runScenario({ name: scenario.name, profile, grade: scenario.grade,
      attempts: [{ files: fix, records: [...piTurn("stop"), settledRecord] }],
      expect: { status: scenario.status as "completed" | "failed", committed: scenario.status === "completed" } });
    assert.equal(run.status, scenario.status, JSON.stringify(run.error));
    assert.equal(gradeCalls, scenario.gradeCalls);
    assert.equal(commits, scenario.status === "completed" ? 1 : 0);
    if (scenario.cause) {
      assert.equal(run.error?.code, "verification_failed");
      assert.equal(run.error?.cause_code, scenario.cause);
      assert.equal(run.grader_cause?.code, scenario.cause);
    }
  });
}
