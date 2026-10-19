import assert from "node:assert/strict";
import { test } from "node:test";

import { awaitUi, piTurn, runScenario, settledRecord, type Scenario } from "./support/scenario.js";

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
  const { run, gradeCalls } = await runScenario({ name: "H0 grader error", profile, grade: [new Error("grader_timeout")],
    attempts: [{ files: fix, records: [...piTurn("stop"), settledRecord] }],
    expect: { status: "failed", committed: false } });
  assert.equal(run.status, "failed");
  assert.equal(gradeCalls, 1);
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

test("S10 repair prompt carries parsed failures in an untrusted block, never host paths", async () => {
  const failed = { status: "failed" as const, checks: [
    { name: "candidate:flake8", status: "failed" as const, exit_code: 1, artifact: "/var/lib/pi-agent/artifacts/run/candidate-flake8.log",
      failures: [{ file: "src/a.js", line: 1, column: 7, code: "E999", message: "unexpected token" }], excerpt: "src/a.js:1:7: E999 unexpected token" },
  ] };
  const { run, prompts } = await runScenario({ name: "S10 structured repair", profile, verify: [failed, "passed"],
    attempts: [
      { files: { "src/a.js": "broken\n" }, records: [...piTurn("stop"), settledRecord] },
      { files: fix, records: [...piTurn("stop"), settledRecord] },
    ],
    expect: { status: "completed", committed: true } });
  assert.equal(run.status, "completed");
  const repair = prompts[1] ?? "";
  assert.match(repair, /src\/a\.js:1:7 E999 unexpected token/);
  const boundary = /<<<(UNTRUSTED-[a-f0-9]+)\n/.exec(repair)?.[1];
  assert.ok(boundary, "diagnostics sit inside a random-boundary block");
  assert.ok(repair.trimEnd().endsWith(`${boundary}>>>`));
  assert.ok(!repair.includes("/var/lib/pi-agent"), "host artifact paths never reach the writer");
});

test("S18 the writer runs checks mid-attempt through the run_checks dialog", async () => {
  const checksRequest = (id: string, toolCallId: string) => ({ type: "extension_ui_request", id, method: "input", title: "pi-agent/run_checks", placeholder: JSON.stringify({ tool_call_id: toolCallId, checks: ["lint"] }) });
  const report = { status: "failed" as const, checks: [{ name: "lint", status: "failed" as const }], report: "src/a.js:1:1 E1 bad" };
  const requested: string[][] = [];
  const { run, responses } = await runScenario({ name: "S18 run_checks", profile,
    quality: {
      runChecks: async request => { requested.push(request.checks ?? []); return report; },
      writerChecks: { maxCalls: 4, checks: ["lint"], defaultChecks: ["lint"], estimateMs: () => 1000, reserveMs: 1000 },
    },
    attempts: [{ files: fix, records: [
      { type: "tool_execution_start", toolCallId: "tc1", toolName: "run_checks", args: { checks: ["lint"] } },
      checksRequest("ui-1", "tc1"), awaitUi,
      { type: "tool_execution_end", toolCallId: "tc1", toolName: "run_checks", result: {}, isError: false },
      checksRequest("ui-forged", "tc1"),
      ...piTurn("stop", "Fixed"), settledRecord,
    ] }],
    expect: { status: "completed", committed: true } });
  assert.equal(run.status, "completed", JSON.stringify(run.error));
  assert.deepEqual(requested, [["lint"]], "a forged request after the tool call ended never reaches the host");
  assert.deepEqual(responses.map(item => item.id), ["ui-1"]);
  assert.deepEqual(JSON.parse((responses[0].response as { value: string }).value), report);
  assert.equal(run.writer_checks_calls, 1);
});
