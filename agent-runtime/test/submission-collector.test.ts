import assert from "node:assert/strict";
import { test } from "node:test";

import { HarnessError, type CauseCode } from "../src/error-taxonomy.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import { collectSubmission, type SubmissionErrors } from "../src/submission-collector.js";

type Step = { record?: Record<string, unknown>; event?: PublicRunEvent };
const start = (args: unknown, toolCallId = "t1"): Step => ({ record: { type: "tool_execution_start", toolName: "submit", toolCallId, args } });
const end = (toolCallId = "t1", isError = false): Step => ({ record: { type: "tool_execution_end", toolName: "submit", toolCallId, isError } });
const submit = (args: unknown, isError = false): Step[] => [start(args), end("t1", isError)];
const settled: Step = { event: { type: "settled", outcome: "completed" } as PublicRunEvent };
const usage = (cost_usd: number): Step => ({ event: { type: "usage", cost_usd } as PublicRunEvent });

const errors: SubmissionErrors = {
  invalid: reason => new HarnessError("grader_invalid_submission", `rejected: ${reason}`),
  missing: () => new HarnessError("grader_did_not_submit", "no submission"),
  timeout: ms => new HarnessError("grader_timeout", `timed out after ${ms}ms`),
  failed: detail => new HarnessError("grader_execution_failed", `failed (${detail})`),
  aborted: () => new Error("collector_aborted"),
};
const validate = (args: unknown) => {
  if ((args as { ok?: unknown })?.ok !== true) throw new Error("Invalid submission shape");
  return args as { ok: true; n?: number };
};

function collect(steps: Step[], extra: { timeoutMs?: number; signal?: AbortSignal; usage?: (cost: number) => void } = {}) {
  return collectSubmission({
    process: ({ onEvent, onRecord }) => { setImmediate(() => { for (const step of steps) { if (step.record) onRecord(step.record); if (step.event) onEvent(step.event); } }); },
    toolName: "submit", validate, timeoutMs: extra.timeoutMs ?? 1_000, signal: extra.signal ?? new AbortController().signal, errors, usage: extra.usage,
  });
}
const cause = (code: CauseCode, message?: RegExp) => (error: unknown) =>
  error instanceof HarnessError && error.causeCode === code && (!message || message.test(error.message));

test("a valid submission is returned with its cost", async () => {
  const seen: number[] = [];
  const result = await collect([usage(0.25), ...submit({ ok: true, n: 1 }), usage(0.5), settled], { usage: cost => seen.push(cost) });
  assert.deepEqual(result, { value: { ok: true, n: 1 }, cost: 0.75 });
  assert.deepEqual(seen, [0.25, 0.5]);
});

test("an end record with a different toolCallId is not accepted", async () => {
  await assert.rejects(collect([start({ ok: true }, "t1"), end("t2"), settled]), cause("grader_did_not_submit"));
});

test("an end record without a matching start is not accepted", async () => {
  await assert.rejects(collect([end("t1"), settled]), cause("grader_did_not_submit"));
});

test("a call to another tool is ignored", async () => {
  await assert.rejects(collect([{ record: { type: "tool_execution_start", toolName: "read", toolCallId: "t1", args: { ok: true } } }, end("t1"), settled]), cause("grader_did_not_submit"));
});

test("an invalid then a valid submission accepts the valid one", async () => {
  const result = await collect([...submit({ ok: false }, true), start({ ok: true, n: 2 }, "t2"), end("t2"), settled]);
  assert.deepEqual(result.value, { ok: true, n: 2 });
});

test("an invalid submission keeps its validation error", async () => {
  await assert.rejects(collect([...submit({ ok: false }), settled]), cause("grader_invalid_submission", /Invalid submission shape/));
});

test("a submission the tool rejects records the rejection as the invalid reason", async () => {
  await assert.rejects(collect([...submit({ ok: true }, true), settled]), cause("grader_invalid_submission", /submit rejected the submission/));
});

test("settling without a submission reports did-not-submit", async () => {
  await assert.rejects(collect([settled]), cause("grader_did_not_submit"));
});

test("an execution that never settles reports a timeout", async () => {
  await assert.rejects(collect([], { timeoutMs: 20 }), cause("grader_timeout", /20ms/));
});

test("an execution error reports an execution failure", async () => {
  await assert.rejects(collect([{ event: { type: "error", code: "container_exited", message: "exit 1" } as PublicRunEvent }]), cause("grader_execution_failed", /container_exited/));
});

test("an approval request reports an execution failure", async () => {
  await assert.rejects(collect([{ event: { type: "approval" } as unknown as PublicRunEvent }]), cause("grader_execution_failed", /approval/));
});

test("aborting rejects with the signal's reason", async () => {
  const controller = new AbortController();
  const pending = collect([], { signal: controller.signal });
  const reason = new Error("caller stopped");
  controller.abort(reason);
  await assert.rejects(pending, (error: unknown) => error === reason);
});

test("aborting without an Error reason uses the aborted fallback", async () => {
  const controller = new AbortController();
  const pending = collect([], { signal: controller.signal });
  controller.abort("stop");
  await assert.rejects(pending, (error: unknown) => !(error instanceof HarnessError) && error instanceof Error && error.message === "collector_aborted");
});

test("an already aborted signal rejects before the execution starts", async () => {
  const controller = new AbortController(); const reason = new Error("early");
  controller.abort(reason);
  let started = false;
  await assert.rejects(collectSubmission({ process: () => { started = true; }, toolName: "submit", validate, timeoutMs: 1_000, signal: controller.signal, errors }), (error: unknown) => error === reason);
  assert.equal(started, false);
});

test("settling on a provider error without a submission reports the execution failure, not did-not-submit", async () => {
  const failed: Step = { event: { type: "settled", outcome: "error", stop_reason: "error", error_message: "429 workflow_budget_unavailable_or_exhausted" } as PublicRunEvent };
  await assert.rejects(collect([failed]), cause("grader_execution_failed", /workflow_budget_unavailable_or_exhausted/));
  // A submission made before the error still counts.
  assert.deepEqual((await collect([...submit({ ok: true }), failed])).value, { ok: true });
});
