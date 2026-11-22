import assert from "node:assert/strict";
import test from "node:test";

import { HarnessError } from "../src/error-taxonomy.js";
import { QualityGate, type GateResult, type GraderSampling } from "../src/quality-gate.js";

type Outcome = "passed" | "needs_changes" | "unknown" | Error;
interface Call { signal: AbortSignal; done: boolean; finish(outcome: Outcome): void }

const criteria = (passed: boolean) => [{ id: "correct", passed, ...(passed ? { evidence: "src/a.ts:1 ok" } : {}) }];
const tick = async () => { for (let turn = 0; turn < 5; turn++) await new Promise(resolve => setImmediate(resolve)); };
/** What the gateway's budget refusal looks like once it has come back through a grader. */
const refused = () => new HarnessError("grader_execution_failed", "429 workflow_budget_unavailable_or_exhausted");

/** A gate whose grader executions stay open until the test finishes each one. */
function openGate(sampling: GraderSampling, extra: { room?: (wanted: number) => number; clock?: { now: number } } = {}) {
  const calls: Call[] = [];
  const asked: number[] = [];
  const controller = new AbortController();
  let waiting: Array<() => void> = [];
  const gate = new QualityGate({
    profile: { id: "p", rubric: ["correct"], maxRepairs: 2, grader: sampling },
    ...(extra.clock ? { now: () => extra.clock!.now } : {}),
    ...(extra.room ? { concurrentSamples: (wanted: number) => { asked.push(wanted); return extra.room!(wanted); } } : {}),
    snapshot: async () => ({ hash: "hash", path: "/snapshot" }),
    verify: async () => ({ status: "passed", checks: [] }),
    grade: (_candidate, _verification, input) => new Promise((resolve, reject) => {
      const number = calls.length + 1;
      const call: Call = { signal: input.signal, done: false, finish: outcome => {
        call.done = true;
        if (outcome instanceof Error) reject(outcome);
        else resolve({ status: outcome, candidate_hash: "hash", criteria: criteria(outcome === "passed"),
          findings: outcome === "needs_changes" ? [{ path: "src/a.ts", line: number, severity: "error", reason: `execution ${number}` }] : [] });
      } };
      calls.push(call);
      const wake = waiting; waiting = []; for (const resolveWaiter of wake) resolveWaiter();
    }),
  });
  let settled = false;
  const result: Promise<GateResult> = gate.evaluate({ run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: controller.signal });
  result.then(() => { settled = true; }, () => { settled = true; });
  return {
    calls, asked, controller, result, settled: () => settled,
    /** Resolves once this many grader executions have started. */
    started: async (count: number) => { while (calls.length < count) await new Promise<void>(resolve => waiting.push(resolve)); },
    /** How many executions exist after the gate has had time to start another. */
    count: async () => { await tick(); return calls.length; },
  };
}
const verdicts = (result: GateResult) => result.grade?.sample_verdicts;
const parallel = (aggregate: GraderSampling["aggregate"] = "unanimous"): GraderSampling => ({ samples: 3, aggregate, parallel: true });

test("with parallel sampling the first sample runs alone and the other two run together", async () => {
  const gate = openGate(parallel());
  await gate.started(1);
  assert.equal(await gate.count(), 1, "the first sample decides whether any other is worth its cost");
  gate.calls[0]!.finish("passed");
  await gate.started(3);
  assert.ok(!gate.calls[1]!.done && !gate.calls[2]!.done, "the second and third samples are in flight at once");
  gate.calls[2]!.finish("passed"); await tick();
  assert.equal(gate.settled(), false);
  gate.calls[1]!.finish("passed");
  const result = await gate.result;
  assert.equal(result.action, "pass");
  assert.equal(result.grade?.grader_attempts, 3);
  assert.equal(verdicts(result)?.length, 3);
});

test("a rejecting first sample ends parallel grading after one execution", async () => {
  const gate = openGate(parallel());
  await gate.started(1);
  gate.calls[0]!.finish("needs_changes");
  const result = await gate.result;
  assert.equal(result.grade?.status, "needs_changes");
  assert.equal(gate.calls.length, 1);
});

test("parallel samples are reported by sample number, not by which finished first", async () => {
  const clock = { now: 0 };
  const gate = openGate(parallel(), { clock });
  await gate.started(1);
  clock.now = 1000; gate.calls[0]!.finish("passed");
  await gate.started(3);
  clock.now = 2500; gate.calls[2]!.finish("unknown"); await tick();
  clock.now = 3000; gate.calls[1]!.finish("passed");
  const result = await gate.result;
  assert.deepEqual(verdicts(result), [
    { status: "passed", grader_attempts: 1 },
    { status: "passed", grader_attempts: 1 },
    { status: "unknown", grader_attempts: 1, cause_code: "insufficient_evidence" },
  ]);
  // The stage takes the first sample plus the slower of the other two, not the sum of all three.
  assert.equal(result.timing?.grade_ms, 3000);
  assert.deepEqual(result.timing?.grade_samples_ms, [1000, 2000, 1500]);
});

test("a rejection does not cancel the sample still running, and that sample gets no retry", async () => {
  const gate = openGate(parallel());
  await gate.started(1);
  gate.calls[0]!.finish("passed");
  await gate.started(3);
  gate.calls[1]!.finish("needs_changes"); await tick();
  // An abandoned request keeps its whole budget reservation, so the third sample is left to finish.
  assert.equal(gate.calls[2]!.signal.aborted, false);
  assert.equal(gate.settled(), false);
  gate.calls[2]!.finish(new HarnessError("grader_timeout", "slow"));
  const result = await gate.result;
  assert.equal(gate.calls.length, 3, "the outcome is already decided: no fresh grader for the failed sample");
  assert.equal(result.action, "repair");
  assert.equal(result.grade?.status, "needs_changes");
  assert.deepEqual(result.grade?.findings, [{ path: "src/a.ts", line: 2, severity: "error", reason: "execution 2" }]);
  assert.deepEqual(verdicts(result), [
    { status: "passed", grader_attempts: 1 },
    { status: "needs_changes", grader_attempts: 1 },
    { status: "unknown", grader_attempts: 1, cause_code: "grader_timeout" },
  ]);
});

test("each parallel sample keeps its own retry", async () => {
  const gate = openGate(parallel());
  await gate.started(1);
  gate.calls[0]!.finish("passed");
  await gate.started(3);
  gate.calls[1]!.finish(new Error("grader crashed"));
  await gate.started(4);
  assert.equal(gate.calls[2]!.done, false, "the second sample retries while the third is still running");
  gate.calls[3]!.finish("passed");
  gate.calls[2]!.finish("passed");
  const result = await gate.result;
  assert.equal(result.grade?.status, "passed");
  assert.deepEqual(verdicts(result), [
    { status: "passed", grader_attempts: 1 },
    { status: "passed", grader_attempts: 2 },
    { status: "passed", grader_attempts: 1 },
  ]);
  assert.equal(gate.calls.length, 4);
});

test("an exhausted budget in one parallel sample stops the other and is thrown once both have ended", async () => {
  const gate = openGate(parallel());
  await gate.started(1);
  gate.calls[0]!.finish("passed");
  await gate.started(3);
  gate.calls[1]!.finish(new Error("workflow_budget_exhausted")); await tick();
  assert.equal(gate.calls[2]!.signal.aborted, true);
  assert.equal(gate.settled(), false, "the gate waits for the sample that is still stopping");
  gate.calls[2]!.finish(new Error("grader_aborted"));
  await assert.rejects(gate.result, /workflow_budget_exhausted/);
  assert.equal(gate.calls.length, 3);
});

test("aborting the run stops both parallel samples", async () => {
  const gate = openGate(parallel());
  await gate.started(1);
  gate.calls[0]!.finish("passed");
  await gate.started(3);
  gate.controller.abort(new Error("run cancelled"));
  assert.ok(gate.calls[1]!.signal.aborted && gate.calls[2]!.signal.aborted);
  gate.calls[1]!.finish(new Error("run cancelled")); await tick();
  assert.equal(gate.settled(), false);
  gate.calls[2]!.finish(new Error("run cancelled"));
  await assert.rejects(gate.result, /run cancelled/);
  assert.equal(gate.calls.length, 3);
});

test("a sample the gateway refused for budget while its sibling ran is graded again on its own", async () => {
  const gate = openGate(parallel());
  await gate.started(1);
  gate.calls[0]!.finish("passed");
  await gate.started(3);
  gate.calls[1]!.finish(refused());
  // Its sibling still holds a reservation: a fresh grader now would be refused as well.
  assert.equal(await gate.count(), 3);
  gate.calls[2]!.finish("passed");
  await gate.started(4);
  assert.equal(gate.calls[3]!.signal.aborted, false);
  gate.calls[3]!.finish("passed");
  const result = await gate.result;
  assert.equal(result.action, "pass");
  assert.deepEqual(verdicts(result), [
    { status: "passed", grader_attempts: 1 },
    { status: "passed", grader_attempts: 2 },
    { status: "passed", grader_attempts: 1 },
  ]);
  assert.equal(result.grade?.grader_attempts, 4);
});

test("a sample that timed out beside a sibling is graded again once, without a further retry", async () => {
  const clock = { now: 0 };
  const gate = openGate(parallel(), { clock });
  await gate.started(1);
  gate.calls[0]!.finish("passed");
  await gate.started(3);
  // The grader's timer runs while it queues for a container, so a timeout here can be the sibling's doing.
  gate.calls[2]!.finish(new HarnessError("grader_timeout", "slow"));
  await gate.started(4);
  gate.calls[3]!.finish(new HarnessError("grader_timeout", "slow")); await tick();
  clock.now = 1000; gate.calls[1]!.finish("passed");
  await gate.started(5);
  clock.now = 1400; gate.calls[4]!.finish(new HarnessError("grader_timeout", "slow"));
  const result = await gate.result;
  assert.equal(gate.calls.length, 5, "the solo attempt is the last one");
  assert.equal(result.grade?.status, "unknown");
  assert.equal(result.grade?.cause?.code, "grader_timeout");
  assert.deepEqual(verdicts(result)?.[2], { status: "unknown", grader_attempts: 3, cause_code: "grader_timeout" });
  // The solo attempt's time is added to the sample's own.
  assert.equal(result.timing?.grade_samples_ms?.[2], 400);
});

test("a failed parallel sample is not graded again once the others decide the outcome", async () => {
  const gate = openGate(parallel());
  await gate.started(1);
  gate.calls[0]!.finish("passed");
  await gate.started(3);
  gate.calls[2]!.finish("needs_changes"); await tick();
  gate.calls[1]!.finish(refused());
  const result = await gate.result;
  assert.equal(gate.calls.length, 3);
  assert.equal(result.grade?.status, "needs_changes");
  assert.deepEqual(verdicts(result)?.map(sample => sample.status), ["passed", "unknown", "needs_changes"]);
});

test("parallel majority sampling runs two samples together and a third only on a split", async () => {
  const agreed = openGate(parallel("majority"));
  await agreed.started(2);
  assert.equal(await agreed.count(), 2, "two samples can already decide a majority of three");
  agreed.calls[1]!.finish("passed"); agreed.calls[0]!.finish("passed");
  assert.equal((await agreed.result).grade?.status, "passed");
  assert.equal(agreed.calls.length, 2);

  const split = openGate(parallel("majority"));
  await split.started(2);
  split.calls[0]!.finish("passed"); split.calls[1]!.finish("needs_changes");
  await split.started(3);
  split.calls[2]!.finish("needs_changes");
  const result = await split.result;
  assert.equal(result.grade?.status, "needs_changes");
  assert.deepEqual(verdicts(result)?.map(sample => sample.status), ["passed", "needs_changes", "needs_changes"]);
});

test("parallel sampling runs one sample at a time when there is room for only one", async () => {
  const gate = openGate(parallel(), { room: () => 1 });
  await gate.started(1);
  gate.calls[0]!.finish("passed");
  await gate.started(2);
  assert.equal(await gate.count(), 2);
  gate.calls[1]!.finish("passed");
  await gate.started(3);
  gate.calls[2]!.finish("passed");
  assert.equal((await gate.result).grade?.status, "passed");
  // Asked once for the two samples left, then nothing to ask about a single one.
  assert.deepEqual(gate.asked, [2]);
});

test("without the parallel option samples still run strictly one after another", async () => {
  const gate = openGate({ samples: 3, aggregate: "unanimous" }, { room: () => 3 });
  await gate.started(1);
  gate.calls[0]!.finish("passed");
  await gate.started(2);
  assert.equal(await gate.count(), 2);
  gate.calls[1]!.finish("needs_changes");
  const result = await gate.result;
  assert.equal(gate.calls.length, 2);
  assert.deepEqual(verdicts(result)?.map(sample => sample.status), ["passed", "needs_changes"]);
  assert.deepEqual(gate.asked, []);
});

test("after an unknown first sample the others run one at a time, since only a rejection can still decide", async () => {
  const gate = openGate(parallel());
  await gate.started(1);
  gate.calls[0]!.finish("unknown");
  await gate.started(2);
  assert.equal(await gate.count(), 2, "a rejection from the second sample would make the third pointless");
  gate.calls[1]!.finish("needs_changes");
  const result = await gate.result;
  assert.equal(gate.calls.length, 2);
  assert.equal(result.grade?.status, "needs_changes");
});

test("parallel sampling grades one at a time when the budget room cannot be read", async () => {
  const gate = openGate(parallel(), { room: () => { throw new Error("database is closed"); } });
  await gate.started(1);
  gate.calls[0]!.finish("passed");
  await gate.started(2);
  assert.equal(await gate.count(), 2);
  gate.calls[1]!.finish("passed");
  await gate.started(3);
  gate.calls[2]!.finish("passed");
  assert.equal((await gate.result).action, "pass");
});
