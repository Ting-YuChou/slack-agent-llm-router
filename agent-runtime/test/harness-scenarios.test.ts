import assert from "node:assert/strict";
import { test } from "node:test";

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
