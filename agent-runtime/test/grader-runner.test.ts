import assert from "node:assert/strict";
import { test } from "node:test";

import { HarnessError } from "../src/error-taxonomy.js";
import { GraderRunner, type GraderProcessFactory } from "../src/grader.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";

const candidate = { hash: "h1", path: "/snapshot", patch: "diff" };
const rubric = ["correct"];
const valid = { status: "passed", candidate_hash: "h1", criteria: [{ id: "correct", passed: true, evidence: "src/a.js:1 returns 2" }], findings: [] };
const inconsistent = { ...valid, criteria: [{ id: "correct", passed: false, evidence: "" }] };

type Step = { record?: Record<string, unknown>; event?: PublicRunEvent };
const submit = (args: unknown, isError = false): Step[] => [
  { record: { type: "tool_execution_start", toolName: "submit_grade", toolCallId: "t1", args } },
  { record: { type: "tool_execution_end", toolName: "submit_grade", toolCallId: "t1", isError } },
];
const settled: Step = { event: { type: "settled", outcome: "completed" } as PublicRunEvent };

function scripted(steps: Step[]): GraderProcessFactory {
  return (_options, onEvent, onRecord) => ({
    prompt: () => { setImmediate(() => { for (const step of steps) { if (step.record) onRecord(step.record); if (step.event) onEvent(step.event); } }); },
    close: async () => {},
  });
}

function runner(steps: Step[], extra: Partial<ConstructorParameters<typeof GraderRunner>[0]> = {}) {
  return new GraderRunner({ image: "img", network: "net", gatewayUrl: "http://gw", stateRoot: `${process.env.TMPDIR ?? "/tmp"}/grader-runner-test`, token: () => "t", createProcess: scripted(steps), ...extra });
}

const grade = (r: GraderRunner, signal = new AbortController().signal) => r.grade(candidate, { status: "passed", checks: [] }, "fix", rubric, "parent", signal);

test("a valid submission is returned", async () => {
  const result = await grade(runner([...submit(valid), settled]));
  assert.equal(result.status, "passed");
});

test("settling without a submission reports grader_did_not_submit", async () => {
  await assert.rejects(grade(runner([settled])), (error: unknown) => error instanceof HarnessError && error.causeCode === "grader_did_not_submit");
});

test("an invalid submission keeps its validation error", async () => {
  await assert.rejects(grade(runner([...submit(inconsistent, true), settled])),
    (error: unknown) => error instanceof HarnessError && error.causeCode === "grader_invalid_submission" && /Inconsistent grader verdict/.test(error.message));
});

test("execution errors report grader_execution_failed", async () => {
  await assert.rejects(grade(runner([{ event: { type: "error", code: "container_exited", message: "exit 1" } }])),
    (error: unknown) => error instanceof HarnessError && error.causeCode === "grader_execution_failed" && /container_exited/.test(error.message));
});

test("a grader that never settles reports grader_timeout", async () => {
  await assert.rejects(grade(runner([], { timeoutMs: 20 })), (error: unknown) => error instanceof HarnessError && error.causeCode === "grader_timeout");
});

test("aborting rejects with the abort, not a grader cause", async () => {
  const controller = new AbortController();
  const pending = grade(runner([]), controller.signal);
  controller.abort();
  await assert.rejects(pending, (error: unknown) => !(error instanceof HarnessError));
});
